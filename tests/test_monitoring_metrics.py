from fastapi.testclient import TestClient

from script_agent.api import app as api_module


def test_metrics_endpoint_exposes_http_and_session_metrics(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    headers = {"X-Tenant-Id": "tenant_dev", "X-Role": "admin", "X-User-Id": "metrics-u1"}
    create = client.post(
        "/api/v1/sessions",
        json={"influencer_name": "监控测试", "category": "食品"},
        headers=headers,
    )
    assert create.status_code == 200

    health = client.get("/api/v1/health")
    assert health.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    assert "script_agent_http_requests_total" in body
    assert "script_agent_http_request_duration_seconds" in body
    assert "script_agent_http_inflight_requests" in body
    assert "script_agent_active_sessions" in body
    assert "script_agent_session_events_total" in body
