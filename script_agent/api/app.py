"""
API层 - FastAPI应用

端点:
  POST /api/v1/sessions           创建会话
  POST /api/v1/generate           生成话术 (同步)
  POST /api/v1/generate/stream    生成话术 (流式SSE)
  GET  /api/v1/sessions           列出会话
  GET  /api/v1/sessions/{id}      获取会话详情
  GET  /api/v1/health             健康检查

认证: X-API-Key 或 Authorization: Bearer <jwt>
租户隔离: 从认证信息中提取 tenant_id, 限制数据访问范围
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from script_agent.agents.orchestrator import Orchestrator
from script_agent.api.auth import (
    AuthContext,
    authenticate_user,
    get_auth_context,
    issue_access_token,
    register_user,
)
from script_agent.observability import metrics as obs
from script_agent.services.checkpoint_store import WorkflowCheckpointManager
from script_agent.services.core_rate_limiter import CoreRateLimiter
from script_agent.services.concurrency import (
    create_session_lock_manager,
    SessionLockTimeoutError,
)
from script_agent.services.session_manager import SessionManager
from script_agent.config.settings import settings

logger = logging.getLogger(__name__)


# ===================================================================
# Lifespan (startup / shutdown)
# ===================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up")
    obs.set_app_info(settings.app_name, app.version, settings.env)
    yield
    logger.info("Application shutting down")
    await orchestrator.shutdown()
    await session_lock_manager.close()
    await checkpoint_manager.close()
    await core_rate_limiter.close()
    await session_manager.close()


# ===================================================================
# FastAPI App
# ===================================================================

app = FastAPI(
    title="话术助手 Agent API",
    description="基于LoRA微调的多Agent协作系统，为达人自动生成垂类定制话术",
    version="1.2.0",
    lifespan=lifespan,
)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

# 全局单例
orchestrator = Orchestrator()
session_manager = SessionManager()
session_lock_manager = create_session_lock_manager()
checkpoint_manager = WorkflowCheckpointManager()
core_rate_limiter = CoreRateLimiter()


def _resolve_metrics_path(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", "")
    if route_path:
        return route_path
    return request.url.path


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    path = _resolve_metrics_path(request)
    method = request.method
    status_code = 500
    start = time.perf_counter()
    obs.inc_http_inflight()
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        obs.dec_http_inflight()
        # 避免 Prometheus 抓取本身淹没业务流量指标。
        if path != "/metrics":
            elapsed = time.perf_counter() - start
            obs.observe_http_request(method, path, status_code, elapsed)


# ===================================================================
# Request / Response Models
# ===================================================================

class CreateSessionRequest(BaseModel):
    influencer_id: str = ""
    influencer_name: str = ""
    category: str = ""


class CreateSessionResponse(BaseModel):
    session_id: str
    message: str = "会话创建成功"


class GenerateRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    query: str = Field(..., description="用户输入", min_length=1)
    trace_id: Optional[str] = None


class GenerateResponse(BaseModel):
    success: bool
    trace_id: str = ""
    script_content: str = ""
    quality_score: float = 0.0
    intent: str = ""
    confidence: float = 0.0
    clarification_needed: bool = False
    clarification_question: str = ""
    timing_ms: dict = {}
    error: str = ""


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    tenant_id: str = Field(default="tenant_dev", min_length=1, max_length=64)
    role: str = Field(default="user", min_length=1, max_length=16)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400
    username: str = ""
    tenant_id: str = ""
    role: str = "user"


# ===================================================================
# 租户隔离辅助
# ===================================================================

async def _load_session_with_tenant_check(
    session_id: str, auth: AuthContext
):
    """加载会话并校验租户 + 用户归属"""
    session = await session_manager.load(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    if session.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")
    # 用户级隔离: 非服务身份(如 JWT/dev_bypass/API key + X-User-Id)不可互看历史。
    current_user_id = _resolve_owner_scope(auth)
    if current_user_id:
        owner_user_id = str(getattr(session, "owner_user_id", "") or "").strip()
        if not owner_user_id:
            raise HTTPException(
                status_code=403,
                detail="该会话未绑定用户归属，无法跨用户访问，请新建会话",
            )
        if owner_user_id != current_user_id:
            raise HTTPException(status_code=403, detail="无权访问其他用户会话")
    return session


def _resolve_owner_scope(auth: AuthContext) -> str:
    user_id = str(getattr(auth, "user_id", "") or "").strip()
    # service:* 视为系统服务身份，不做用户级过滤。
    if not user_id or user_id.startswith("service:"):
        return ""
    return user_id


def _estimate_token_cost(query: str, include_output_budget: bool = False) -> int:
    # 中文粗略估算: 1 token ~= 1.5 字符
    input_tokens = max(1, int(len(query) / 1.5))
    if include_output_budget:
        return input_tokens + int(settings.llm.max_tokens * 0.6)
    return input_tokens


def _format_sse_data(payload: str) -> str:
    """
    按 SSE 规范编码数据:
      - 每一行都以 `data: ` 前缀输出
      - 事件间使用空行分隔
    """
    text = str(payload or "")
    if text == "":
        return "data: \n\n"
    lines = text.splitlines()
    if not lines:
        return "data: \n\n"
    return "".join(f"data: {line}\n" for line in lines) + "\n"


async def _enforce_core_limits(
    auth: AuthContext,
    query: str,
    include_output_budget: bool = False,
) -> None:
    decision = await core_rate_limiter.check_and_consume(
        tenant_id=auth.tenant_id,
        token_cost=_estimate_token_cost(query, include_output_budget),
    )
    if decision.allowed:
        return

    obs.record_request("core_limit", "rate_limited", "none")
    raise HTTPException(
        status_code=429,
        detail={
            "message": "请求超过限流配额",
            "reason": decision.reason,
            "retry_after_seconds": round(decision.retry_after_seconds, 3),
        },
    )


# ===================================================================
# Endpoints
# ===================================================================


@app.get("/", include_in_schema=False)
async def frontend_home():
    """前端交互页面"""
    if not WEB_DIR.exists():
        raise HTTPException(status_code=404, detail="frontend page not found")
    return FileResponse(WEB_DIR / "index.html")


@app.get("/auth", include_in_schema=False)
async def frontend_auth():
    """登录注册页面"""
    if not WEB_DIR.exists():
        raise HTTPException(status_code=404, detail="frontend page not found")
    page = WEB_DIR / "auth.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="auth page not found")
    return FileResponse(page)


@app.get("/api/v1/frontend-config")
async def frontend_config():
    """前端运行时配置（是否暴露连接配置、是否需要登录）。"""
    app_env = os.getenv("APP_ENV", settings.env or "development").lower()
    expose_connection_panel = app_env != "production"
    raw_override = os.getenv("FRONTEND_EXPOSE_CONNECTION_PANEL", "").strip().lower()
    if raw_override in ("true", "false"):
        expose_connection_panel = raw_override == "true"

    registration_enabled = (
        os.getenv("AUTH_REGISTRATION_ENABLED", "true").strip().lower() == "true"
    )
    return {
        "env": app_env,
        "auth_required": app_env != "development",
        "expose_connection_panel": expose_connection_panel,
        "registration_enabled": registration_enabled,
    }


@app.post("/api/v1/auth/register")
async def auth_register(req: RegisterRequest):
    """账号注册（本地 SQLite 用户库）。"""
    registration_enabled = (
        os.getenv("AUTH_REGISTRATION_ENABLED", "true").strip().lower() == "true"
    )
    if not registration_enabled:
        raise HTTPException(status_code=403, detail="Registration disabled")
    ok, msg = register_user(
        username=req.username,
        password=req.password,
        tenant_id=req.tenant_id,
        role=req.role,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": "注册成功"}


@app.post("/api/v1/auth/login", response_model=LoginResponse)
async def auth_login(req: LoginRequest):
    """账号登录并签发 Bearer Token。"""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    expires_in = int(os.getenv("AUTH_TOKEN_EXPIRES_SECONDS", "86400"))
    token = issue_access_token(
        username=user["username"],
        tenant_id=user["tenant_id"],
        role=user["role"],
        expires_in_seconds=expires_in,
    )
    return LoginResponse(
        access_token=token,
        expires_in=expires_in,
        username=user["username"],
        tenant_id=user["tenant_id"],
        role=user["role"],
    )


@app.post("/api/v1/sessions", response_model=CreateSessionResponse)
async def create_session(
    req: CreateSessionRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """创建新会话 — tenant_id 自动从认证信息注入"""
    owner_user_id = _resolve_owner_scope(auth)
    session = await session_manager.create(
        tenant_id=auth.tenant_id,
        owner_user_id=owner_user_id,
        influencer_id=req.influencer_id,
        influencer_name=req.influencer_name,
        category=req.category,
    )
    return CreateSessionResponse(session_id=session.session_id)


@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_script(
    req: GenerateRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """生成话术 (同步)"""
    await _enforce_core_limits(auth, req.query, include_output_budget=True)

    try:
        async with session_lock_manager.acquire(req.session_id):
            session = await _load_session_with_tenant_check(req.session_id, auth)
            session.workflow_snapshot = {
                **(session.workflow_snapshot or {}),
                "actor_role": auth.role,
            }
            result = await orchestrator.handle_request(
                query=req.query,
                session=session,
                trace_id=req.trace_id,
                checkpoint_saver=session_manager.save,
                checkpoint_loader=checkpoint_manager.latest_payload,
                checkpoint_writer=checkpoint_manager.write,
            )
            await session_manager.save(session)
    except SessionLockTimeoutError:
        obs.record_lock_timeout("/api/v1/generate")
        raise HTTPException(
            status_code=409,
            detail="会话正在处理中，请稍后重试",
        )

    resp = GenerateResponse(
        success=result.get("success", False),
        trace_id=result.get("trace_id", ""),
        timing_ms=result.get("timing", {}),
        clarification_needed=result.get("clarification_needed", False),
        clarification_question=result.get("clarification_question", ""),
        error=result.get("error", ""),
    )

    script = result.get("script")
    if script:
        resp.script_content = script.content
        resp.quality_score = script.quality_score

    intent = result.get("intent")
    if intent:
        resp.intent = intent.intent
        resp.confidence = intent.confidence

    return resp


@app.post("/api/v1/generate/stream")
async def generate_script_stream(
    req: GenerateRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """生成话术 (流式SSE)"""
    await _enforce_core_limits(auth, req.query, include_output_budget=True)

    # 先做一次显式鉴权校验，避免在流中抛出 403/404
    await _load_session_with_tenant_check(req.session_id, auth)

    async def event_generator():
        try:
            async with session_lock_manager.acquire(req.session_id):
                session = await _load_session_with_tenant_check(req.session_id, auth)
                session.workflow_snapshot = {
                    **(session.workflow_snapshot or {}),
                    "actor_role": auth.role,
                }
                async for token in orchestrator.handle_stream(
                    query=req.query,
                    session=session,
                    trace_id=req.trace_id,
                    checkpoint_saver=session_manager.save,
                    checkpoint_loader=checkpoint_manager.latest_payload,
                    checkpoint_writer=checkpoint_manager.write,
                ):
                    yield _format_sse_data(token)

                await session_manager.save(session)
                yield _format_sse_data("[DONE]")
        except SessionLockTimeoutError:
            obs.record_lock_timeout("/api/v1/generate/stream")
            yield _format_sse_data("[ERROR] 会话正在处理中，请稍后重试")
        except Exception as exc:
            logger.error("Stream generate failed: %s", exc, exc_info=True)
            yield _format_sse_data("[ERROR] 生成失败，请稍后重试")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.get("/api/v1/sessions")
async def list_sessions(auth: AuthContext = Depends(get_auth_context)):
    """列出当前租户的会话"""
    owner_user_id = _resolve_owner_scope(auth)
    return await session_manager.list_sessions(
        tenant_id=auth.tenant_id,
        owner_user_id=owner_user_id,
    )


@app.get("/api/v1/sessions/{session_id}")
async def get_session(
    session_id: str,
    auth: AuthContext = Depends(get_auth_context),
):
    """获取会话详情"""
    session = await _load_session_with_tenant_check(session_id, auth)
    return {
        "session_id": session.session_id,
        "tenant_id": session.tenant_id,
        "influencer_name": session.influencer_name,
        "category": session.category,
        "turn_count": len(session.turns),
        "turns": [
            {
                "index": t.turn_index,
                "user": t.user_message,
                "assistant": t.assistant_message[:200],
                "has_script": t.generated_script is not None,
            }
            for t in session.turns
        ],
        "generated_scripts_count": len(session.generated_scripts),
    }


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    auth: AuthContext = Depends(get_auth_context),
):
    """删除会话历史（租户隔离）。"""
    await _load_session_with_tenant_check(session_id, auth)
    await session_manager.delete(session_id)
    return {"success": True, "session_id": session_id}


@app.get("/api/v1/health")
async def health_check():
    """健康检查 (无需认证)"""
    lock_stats = await session_lock_manager.stats()
    checkpoint_stats = await checkpoint_manager.stats()
    core_rate_stats = await core_rate_limiter.stats()
    memory_stats = await orchestrator.memory_retriever.stats()
    return {
        "status": "healthy",
        "app": settings.app_name,
        "env": settings.env,
        "orchestration": orchestrator.info(),
        "session_locks": lock_stats,
        "checkpoint_store": checkpoint_stats,
        "core_rate_limit": core_rate_stats,
        "longterm_memory": memory_stats,
    }


@app.get("/api/v1/skills")
async def list_skills(auth: AuthContext = Depends(get_auth_context)):
    """列出所有可用 Skill"""
    return orchestrator.skill_registry.list_skills()


@app.get("/api/v1/sessions/{session_id}/checkpoints")
async def list_session_checkpoints(
    session_id: str,
    limit: int = 20,
    auth: AuthContext = Depends(get_auth_context),
):
    """查看会话 checkpoint 历史（审计）"""
    await _load_session_with_tenant_check(session_id, auth)
    records = await checkpoint_manager.history(session_id, limit=limit)
    return [
        {
            "version": r.get("version"),
            "trace_id": r.get("trace_id"),
            "status": r.get("status"),
            "created_at": r.get("created_at"),
            "checksum": r.get("checksum"),
            "current_state": r.get("payload", {}).get("current_state"),
        }
        for r in records
    ]


@app.get("/api/v1/sessions/{session_id}/checkpoints/latest")
async def get_latest_checkpoint(
    session_id: str,
    auth: AuthContext = Depends(get_auth_context),
):
    """获取最新 checkpoint（含 payload）"""
    await _load_session_with_tenant_check(session_id, auth)
    record = await checkpoint_manager.latest_record(session_id)
    if not record:
        raise HTTPException(status_code=404, detail="checkpoint 不存在")
    return record


@app.get("/api/v1/sessions/{session_id}/checkpoints/{version}")
async def replay_checkpoint_version(
    session_id: str,
    version: int,
    auth: AuthContext = Depends(get_auth_context),
):
    """按版本回放 checkpoint（含 payload）"""
    await _load_session_with_tenant_check(session_id, auth)
    payload = await checkpoint_manager.replay(session_id, version)
    if payload is None:
        raise HTTPException(status_code=404, detail="checkpoint version 不存在")
    return {
        "session_id": session_id,
        "version": version,
        "payload": payload,
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标端点"""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=501,
            detail="prometheus_client not installed",
        )
