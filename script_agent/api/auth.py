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
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

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

    def validate(self, token: str) -> Optional[Dict]:
        """验证 JWT, 返回 payload"""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            # 解码 payload
            payload_b64 = parts[1]
            # 补齐 padding
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            import base64
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            payload = json.loads(payload_bytes)

            # 检查过期
            if payload.get("exp", 0) < time.time():
                logger.warning("JWT expired")
                return None

            # 验证签名
            signing_input = f"{parts[0]}.{parts[1]}".encode()
            expected_sig = hmac.new(
                self._secret, signing_input, hashlib.sha256
            ).hexdigest()

            import base64
            actual_sig_bytes = base64.urlsafe_b64decode(
                parts[2] + "=" * (4 - len(parts[2]) % 4)
            )
            actual_sig = actual_sig_bytes.hex()

            if not hmac.compare_digest(expected_sig, actual_sig):
                logger.warning("JWT signature mismatch")
                return None

            return payload

        except Exception as e:
            logger.warning(f"JWT validation failed: {e}")
            return None


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
_rate_limiter = RateLimiter()
_bearer_scheme = HTTPBearer(auto_error=False)


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
    import os

    # 开发模式跳过认证
    if os.getenv("APP_ENV", "development") == "development":
        tenant_id = request.headers.get("X-Tenant-Id", "tenant_dev")
        return AuthContext(
            tenant_id=tenant_id,
            auth_method="dev_bypass",
        )

    # 方式1: API Key
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
            tenant_info=tenant,
        )

    # 方式2: JWT Bearer
    if credentials:
        payload = _jwt_validator.validate(credentials.credentials)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        tenant_id = payload.get("tenant_id", "")
        if not tenant_id:
            raise HTTPException(status_code=401, detail="Missing tenant_id in token")
        if not _rate_limiter.check(tenant_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return AuthContext(
            tenant_id=tenant_id,
            auth_method="jwt",
        )

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Use X-API-Key or Authorization: Bearer <token>",
    )
