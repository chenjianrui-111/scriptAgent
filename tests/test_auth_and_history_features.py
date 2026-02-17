import uuid

from fastapi.testclient import TestClient

from script_agent.api import app as api_module


def _headers(tenant: str = "tenant_dev", role: str = "admin", bearer: str = ""):
    h = {"X-Tenant-Id": tenant, "X-Role": role}
    if bearer:
        h["Authorization"] = f"Bearer {bearer}"
    return h


def test_delete_session_history_endpoint():
    client = TestClient(api_module.app)
    create = client.post(
        "/api/v1/sessions",
        json={"influencer_name": "删除测试", "category": "美妆"},
        headers=_headers(),
    )
    assert create.status_code == 200
    session_id = create.json()["session_id"]

    delete_resp = client.delete(
        f"/api/v1/sessions/{session_id}",
        headers=_headers(),
    )
    assert delete_resp.status_code == 200
    assert delete_resp.json().get("success") is True

    get_after_delete = client.get(
        f"/api/v1/sessions/{session_id}",
        headers=_headers(),
    )
    assert get_after_delete.status_code == 404


def test_register_login_and_bearer_access_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    client = TestClient(api_module.app)

    username = f"user_{uuid.uuid4().hex[:10]}"
    password = "Passw0rd123"
    tenant = "tenant_auth_test"

    register_resp = client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "password": password,
            "tenant_id": tenant,
            "role": "user",
        },
    )
    assert register_resp.status_code == 200

    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert login_resp.status_code == 200
    token = login_resp.json().get("access_token", "")
    assert token

    create_session_resp = client.post(
        "/api/v1/sessions",
        json={"influencer_name": "JWT测试", "category": "食品"},
        headers=_headers(bearer=token),
    )
    assert create_session_resp.status_code == 200
    assert create_session_resp.json().get("session_id")


def test_frontend_config_hides_connection_panel_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("FRONTEND_EXPOSE_CONNECTION_PANEL", raising=False)
    client = TestClient(api_module.app)
    resp = client.get("/api/v1/frontend-config")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("auth_required") is True
    assert payload.get("expose_connection_panel") is False


def test_frontend_assets_include_auth_page_and_delete_action():
    client = TestClient(api_module.app)
    auth_page = client.get("/auth")
    assert auth_page.status_code == 200
    assert 'id="loginForm"' in auth_page.text
    assert 'id="registerForm"' in auth_page.text

    js = client.get("/web/app.js")
    assert js.status_code == 200
    assert "async function deleteSession" in js.text
    assert "authActionBtn" in js.text
