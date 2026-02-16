from fastapi.testclient import TestClient

from script_agent.api import app as api_module
from script_agent.models.message import GeneratedScript, IntentResult


def _headers(tenant: str = "tenant_dev", role: str = "admin", api_key: str = ""):
    h = {"X-Tenant-Id": tenant, "X-Role": role}
    if api_key:
        h["X-API-Key"] = api_key
    return h


def test_frontend_assets_and_contract(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    home = client.get("/")
    assert home.status_code == 200
    # 核心容器 + 连接配置
    assert 'id="chatArea"' in home.text
    assert 'id="quickReplies"' in home.text
    assert 'id="textInputWrap"' in home.text
    assert 'id="connPanel"' in home.text
    assert 'id="apiBaseUrl"' in home.text
    assert 'id="apiKeyInput"' in home.text
    assert 'id="bearerTokenInput"' in home.text

    js = client.get("/web/app.js")
    assert js.status_code == 200
    # 前端调用能力契约
    assert "function buildApiUrl" in js.text
    assert "function collectHeaders" in js.text
    assert "X-API-Key" in js.text
    assert "Authorization" in js.text
    assert "generateStream" in js.text

    css = client.get("/web/styles.css")
    assert css.status_code == 200
    assert ".conn-panel" in css.text
    assert ".btn-pill" in css.text


def test_frontend_flow_e2e_sync_and_stream(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")

    async def fake_handle_request(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        return {
            "success": True,
            "trace_id": trace_id or "trace-e2e",
            "timing": {"total": 12.5},
            "script": GeneratedScript(content=f"生成结果: {query}", quality_score=0.92),
            "intent": IntentResult(
                intent="script_generation",
                confidence=0.99,
                slots={"category": "美妆", "scenario": "开场"},
            ),
        }

    async def fake_handle_stream(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        for token in ["流式", "输出", "验证"]:
            yield token

    monkeypatch.setattr(api_module.orchestrator, "handle_request", fake_handle_request)
    monkeypatch.setattr(api_module.orchestrator, "handle_stream", fake_handle_stream)

    client = TestClient(api_module.app)

    create_resp = client.post(
        "/api/v1/sessions",
        json={
            "influencer_id": "inf-e2e",
            "influencer_name": "测试达人",
            "category": "美妆",
        },
        headers=_headers(),
    )
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    sync_resp = client.post(
        "/api/v1/generate",
        json={"session_id": session_id, "query": "给我一段美妆开场话术"},
        headers=_headers(),
    )
    assert sync_resp.status_code == 200
    sync_body = sync_resp.json()
    assert sync_body["success"] is True
    assert "生成结果" in sync_body["script_content"]
    assert sync_body["intent"] == "script_generation"

    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": session_id, "query": "继续输出流式话术"},
        headers=_headers(),
    )
    assert stream_resp.status_code == 200
    assert "data: 流式" in stream_resp.text
    assert "data: 输出" in stream_resp.text
    assert "data: 验证" in stream_resp.text
    assert "data: [DONE]" in stream_resp.text


def test_frontend_flow_stream_error_path(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")

    async def broken_handle_stream(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        raise RuntimeError("mock stream crash")
        yield "unused"

    monkeypatch.setattr(api_module.orchestrator, "handle_stream", broken_handle_stream)

    client = TestClient(api_module.app)
    create_resp = client.post(
        "/api/v1/sessions",
        json={"influencer_name": "测试达人", "category": "美妆"},
        headers=_headers(),
    )
    session_id = create_resp.json()["session_id"]

    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": session_id, "query": "测试异常流式"},
        headers=_headers(),
    )
    assert stream_resp.status_code == 200
    assert "data: [ERROR] 生成失败，请稍后重试" in stream_resp.text


def test_frontend_production_auth_with_api_key(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    client = TestClient(api_module.app)

    # 未携带 API Key/Bearer 时，生产环境应拒绝
    unauthorized = client.post(
        "/api/v1/sessions",
        json={"influencer_name": "生产测试", "category": "美妆"},
    )
    assert unauthorized.status_code == 401

    # 携带 API Key 后可正常创建会话
    authorized = client.post(
        "/api/v1/sessions",
        json={"influencer_name": "生产测试", "category": "美妆"},
        headers=_headers(api_key="sk-dev-test-key-001"),
    )
    assert authorized.status_code == 200
    assert authorized.json().get("session_id")


def test_frontend_tenant_isolation(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    create_resp = client.post(
        "/api/v1/sessions",
        json={"influencer_name": "租户A", "category": "美妆"},
        headers=_headers(tenant="tenant_dev"),
    )
    session_id = create_resp.json()["session_id"]

    # 跨租户访问应被拒绝
    forbidden = client.post(
        "/api/v1/generate",
        json={"session_id": session_id, "query": "跨租户尝试"},
        headers=_headers(tenant="tenant_demo"),
    )
    assert forbidden.status_code == 403
