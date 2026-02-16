from fastapi.testclient import TestClient

from script_agent.api import app as api_module
from script_agent.models.message import GeneratedScript, IntentResult


def test_frontend_assets_served(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    home = client.get("/")
    assert home.status_code == 200
    assert "chat-area" in home.text

    js = client.get("/web/app.js")
    assert js.status_code == 200
    assert "advanceFlow" in js.text

    css = client.get("/web/styles.css")
    assert css.status_code == 200
    assert "--bg-primary" in css.text


def test_frontend_flow_e2e(monkeypatch):
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
        headers={"X-Tenant-Id": "tenant_dev", "X-Role": "admin"},
    )
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    sync_resp = client.post(
        "/api/v1/generate",
        json={"session_id": session_id, "query": "给我一段美妆开场话术"},
        headers={"X-Tenant-Id": "tenant_dev", "X-Role": "admin"},
    )
    assert sync_resp.status_code == 200
    sync_body = sync_resp.json()
    assert sync_body["success"] is True
    assert "生成结果" in sync_body["script_content"]
    assert sync_body["intent"] == "script_generation"

    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": session_id, "query": "继续输出流式话术"},
        headers={"X-Tenant-Id": "tenant_dev", "X-Role": "admin"},
    )
    assert stream_resp.status_code == 200
    assert "data: 流式" in stream_resp.text
    assert "data: 输出" in stream_resp.text
    assert "data: 验证" in stream_resp.text
    assert "data: [DONE]" in stream_resp.text
