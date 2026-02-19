"""
认证鉴权中间件

支持两种认证方式:
  1. API Key: Header "X-API-Key" — 适用于服务间调用
  2. JWT Bearer Token: Header "Authorization: Bearer <token>" — 适用于用户请求

租户隔离: 从认证信息中提取 tenant_id, 注入请求上下文
限流: 基于 tenant_id 的令牌桶限流
"""

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
import threading
import base64
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# ======================================================================
#  数据模型
# ======================================================================

@dataclass
class TenantInfo:
    """租户信息"""
    tenant_id: str
    tenant_name: str = ""
    rate_limit: int = 60          # 每分钟最大请求数
    allowed_categories: list = field(default_factory=list)
    is_active: bool = True


@dataclass
class AuthContext:
    """请求认证上下文 — 注入到每个 API handler"""
    tenant_id: str
    auth_method: str              # "api_key" | "jwt"
    role: str = "user"
    user_id: str = ""
    tenant_info: Optional[TenantInfo] = None


# ======================================================================
#  API Key 管理 (生产环境应从数据库/配置中心加载)
# ======================================================================

class APIKeyManager:
    """API Key 验证与租户映射"""

    def __init__(self):
        # 开发环境预置 key (生产环境从 DB 加载)
        self._keys: Dict[str, TenantInfo] = {
            "sk-dev-test-key-001": TenantInfo(
                tenant_id="tenant_dev",
                tenant_name="开发测试租户",
                rate_limit=120,
            ),
            "sk-demo-key-002": TenantInfo(
                tenant_id="tenant_demo",
                tenant_name="演示租户",
                rate_limit=60,
            ),
        }

    def validate(self, api_key: str) -> Optional[TenantInfo]:
        """验证 API Key, 返回租户信息"""
        tenant = self._keys.get(api_key)
        if tenant and tenant.is_active:
            return tenant
        return None

    def register_key(self, api_key: str, tenant: TenantInfo):
        """注册新的 API Key"""
        self._keys[api_key] = tenant


# ======================================================================
#  JWT 验证 (简化版, 生产环境应用 PyJWT + RS256)
# ======================================================================

class JWTValidator:
    """JWT 验证器 (HMAC-SHA256 简化实现)"""

    def __init__(self, secret: str = "script-agent-jwt-secret-change-in-production"):
        self._secret = secret.encode()

    def issue(self, payload: Dict[str, Any], expires_in_seconds: int = 86400) -> str:
        now = int(time.time())
        body = dict(payload or {})
        body.setdefault("iat", now)
        body["exp"] = now + max(1, int(expires_in_seconds))

        header_b64 = self._b64url_encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}, ensure_ascii=False, separators=(",", ":")).encode()
        )
        payload_b64 = self._b64url_encode(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        )
        signing_input = f"{header_b64}.{payload_b64}".encode()
        sig = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        sig_b64 = self._b64url_encode(sig)
        return f"{header_b64}.{payload_b64}.{sig_b64}"

    def validate(self, token: str) -> Optional[Dict]:
        """验证 JWT, 返回 payload"""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            signing_input = f"{parts[0]}.{parts[1]}".encode()
            expected_sig = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
            actual_sig = self._b64url_decode(parts[2])
            if not hmac.compare_digest(expected_sig, actual_sig):
                logger.warning("JWT signature mismatch")
                return None

            payload = json.loads(self._b64url_decode(parts[1]))
            if payload.get("exp", 0) < time.time():
                logger.warning("JWT expired")
                return None
            return payload

        except Exception as e:
            logger.warning(f"JWT validation failed: {e}")
            return None

    def _b64url_encode(self, value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode().rstrip("=")

    def _b64url_decode(self, value: str) -> bytes:
        padding = "=" * ((4 - len(value) % 4) % 4)
        return base64.urlsafe_b64decode(value + padding)


class UserAuthStore:
    """本地用户存储（SQLite）。"""

    def __init__(self, db_path: str = ""):
        self._db_path = db_path or os.getenv("AUTH_DB_PATH", "auth_users.db")
        self._lock = threading.Lock()
        self._ready = False

    def register_user(
        self,
        username: str,
        password: str,
        tenant_id: str,
        role: str = "user",
    ) -> Tuple[bool, str]:
        uname = (username or "").strip()
        tenant = (tenant_id or "").strip() or "tenant_dev"
        role_value = (role or "").strip() or "user"
        if len(uname) < 3:
            return False, "用户名至少3个字符"
        if len(password or "") < 6:
            return False, "密码至少6个字符"
        if role_value not in ("user", "admin", "service"):
            role_value = "user"

        self._ensure_schema()
        pwd_hash, salt = self._hash_password(password)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO auth_users (username, password_hash, password_salt, tenant_id, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (uname, pwd_hash, salt, tenant, role_value, int(time.time())),
                )
                conn.commit()
            return True, "ok"
        except sqlite3.IntegrityError:
            return False, "用户名已存在"
        except Exception as exc:
            logger.error("register user failed: %s", exc)
            return False, "注册失败，请稍后重试"

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, str]]:
        uname = (username or "").strip()
        if not uname or not password:
            return None
        self._ensure_schema()
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT username, password_hash, password_salt, tenant_id, role
                    FROM auth_users
                    WHERE username = ?
                    LIMIT 1
                    """,
                    (uname,),
                ).fetchone()
        except Exception as exc:
            logger.error("authenticate user failed: %s", exc)
            return None

        if row is None:
            return None
        expected_hash = str(row["password_hash"] or "")
        actual_hash, _salt = self._hash_password(password, str(row["password_salt"] or ""))
        if not hmac.compare_digest(expected_hash, actual_hash):
            return None
        return {
            "username": str(row["username"] or ""),
            "tenant_id": str(row["tenant_id"] or "tenant_dev"),
            "role": str(row["role"] or "user"),
        }

    def _ensure_schema(self) -> None:
        with self._lock:
            if self._ready:
                return
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS auth_users (
                        username TEXT PRIMARY KEY,
                        password_hash TEXT NOT NULL,
                        password_salt TEXT NOT NULL,
                        tenant_id TEXT NOT NULL DEFAULT 'tenant_dev',
                        role TEXT NOT NULL DEFAULT 'user',
                        created_at INTEGER NOT NULL
                    )
                    """
                )
                conn.commit()
            self._ready = True

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _hash_password(self, password: str, salt_hex: str = "") -> Tuple[str, str]:
        salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            (password or "").encode(),
            salt,
            150000,
        )
        return digest.hex(), salt.hex()


# ======================================================================
#  限流器 (令牌桶)
# ======================================================================

class RateLimiter:
    """基于租户的令牌桶限流"""

    def __init__(self):
        self._buckets: Dict[str, Dict] = defaultdict(
            lambda: {"tokens": 60, "last_refill": time.time()}
        )

    def check(self, tenant_id: str, max_rpm: int = 60) -> bool:
        """检查是否允许请求, True = 允许"""
        bucket = self._buckets[tenant_id]
        now = time.time()

        # 补充令牌
        elapsed = now - bucket["last_refill"]
        refill = elapsed * (max_rpm / 60.0)
        bucket["tokens"] = min(max_rpm, bucket["tokens"] + refill)
        bucket["last_refill"] = now

        # 消耗令牌
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False


# ======================================================================
#  FastAPI 依赖注入
# ======================================================================

# 全局单例
_api_key_manager = APIKeyManager()
_jwt_validator = JWTValidator()
_user_store = UserAuthStore()
_rate_limiter = RateLimiter()
_bearer_scheme = HTTPBearer(auto_error=False)


def register_user(username: str, password: str, tenant_id: str, role: str = "user") -> Tuple[bool, str]:
    return _user_store.register_user(username, password, tenant_id, role)


def authenticate_user(username: str, password: str) -> Optional[Dict[str, str]]:
    return _user_store.authenticate_user(username, password)


def issue_access_token(
    username: str,
    tenant_id: str,
    role: str,
    expires_in_seconds: int = 86400,
) -> str:
    payload = {
        "sub": username,
        "tenant_id": tenant_id,
        "role": role,
    }
    return _jwt_validator.issue(payload, expires_in_seconds=expires_in_seconds)


async def get_auth_context(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> AuthContext:
    """
    认证依赖 — 从请求中提取认证信息

    支持:
      1. X-API-Key header
      2. Authorization: Bearer <jwt>
      3. 开发模式: 无认证 (APP_ENV=development)
    """
    # 开发模式跳过认证
    if os.getenv("APP_ENV", "development") == "development":
        tenant_id = request.headers.get("X-Tenant-Id", "tenant_dev")
        user_id = request.headers.get("X-User-Id", "dev_user")
        return AuthContext(
            tenant_id=tenant_id,
            auth_method="dev_bypass",
            role=request.headers.get("X-Role", "admin"),
            user_id=user_id,
        )

    # 方式1: JWT Bearer
    if credentials:
        payload = _jwt_validator.validate(credentials.credentials)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        tenant_id = payload.get("tenant_id", "")
        user_id = payload.get("sub", "")
        if not tenant_id:
            raise HTTPException(status_code=401, detail="Missing tenant_id in token")
        if not user_id:
            raise HTTPException(status_code=401, detail="Missing subject in token")
        if not _rate_limiter.check(tenant_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return AuthContext(
            tenant_id=tenant_id,
            auth_method="jwt",
            role=payload.get("role", "user"),
            user_id=user_id,
        )

    # 方式2: API Key (当没有 Bearer 时才使用)
    api_key = request.headers.get("X-API-Key")
    if api_key:
        tenant = _api_key_manager.validate(api_key)
        if not tenant:
            raise HTTPException(status_code=401, detail="Invalid API Key")
        # 限流
        if not _rate_limiter.check(tenant.tenant_id, tenant.rate_limit):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return AuthContext(
            tenant_id=tenant.tenant_id,
            auth_method="api_key",
            role="service",
            user_id=f"service:{tenant.tenant_id}",
            tenant_info=tenant,
        )

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Use X-API-Key or Authorization: Bearer <token>",
    )
