import uuid

from fastapi.testclient import TestClient

from script_agent.api import app as api_module
from script_agent.models.message import GeneratedScript, QualityResult


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


def test_api_e2e_product_switch_generation_contains_new_product(monkeypatch):
    """
    API级多轮E2E:
      /sessions -> /generate(三只松鼠) -> /generate(换品到卫龙辣条)
    要求第二轮文案必须包含新商品名，且保持 script_generation 意图。
    """
    monkeypatch.setenv("APP_ENV", "development")

    async def _noop_enforce(*args, **kwargs):
        return None

    monkeypatch.setattr(api_module, "_enforce_core_limits", _noop_enforce)

    skill = api_module.orchestrator.skill_registry.get("script_generation")
    assert skill is not None

    old_script_agent = skill._script_agent
    old_quality_agent = skill._quality_agent
    old_fast_classify = api_module.orchestrator.intent_agent._fast_classify

    class ProductEchoScriptAgent:
        async def __call__(self, message):
            slots = message.payload.get("slots", {})
            product = message.payload.get("product")
            product_name = getattr(product, "name", "") or slots.get("product_name", "") or "未知商品"
            content = (
                f"今天重点介绍{product_name}，核心卖点清晰、口感层次丰富，"
                "直播间可直接口播转化，并引导观众马上下单锁定福利。"
            )
            script = GeneratedScript(
                content=content,
                category=slots.get("category", "食品"),
                scenario=slots.get("scenario", "直播带货"),
                quality_score=0.92,
            )
            return message.create_response(
                payload={"script": script},
                source="script_generation",
            )

    class PassQualityAgent:
        async def __call__(self, message):
            return message.create_response(
                payload={"quality_result": QualityResult(passed=True, overall_score=0.93)},
                source="quality_check",
            )

    try:
        def _fake_fast_classify(query: str):
            if "换成" in query:
                # 故意模拟误判，验证换品纠偏逻辑会改回 script_generation
                return "script_modification", 0.92
            return "script_generation", 0.95

        api_module.orchestrator.intent_agent._fast_classify = _fake_fast_classify
        skill._script_agent = ProductEchoScriptAgent()
        skill._quality_agent = PassQualityAgent()

        client = TestClient(api_module.app)

        create = client.post(
            "/api/v1/sessions",
            json={"influencer_name": "换品E2E达人", "category": "食品"},
            headers=_headers(),
        )
        assert create.status_code == 200
        session_id = create.json()["session_id"]

        first = client.post(
            "/api/v1/generate",
            json={"session_id": session_id, "query": "请介绍三只松鼠零食的直播卖点话术"},
            headers=_headers(),
        )
        assert first.status_code == 200
        body1 = first.json()
        assert body1["success"] is True
        assert body1["intent"] == "script_generation"
        assert "三只松鼠零食" in body1["script_content"]

        second = client.post(
            "/api/v1/generate",
            json={
                "session_id": session_id,
                "query": "不要三只松鼠零食了，换成卫龙辣条，继续介绍直播卖点话术",
            },
            headers=_headers(),
        )
        assert second.status_code == 200
        body2 = second.json()
        assert body2["success"] is True
        assert body2["intent"] == "script_generation"
        assert "卫龙辣条" in body2["script_content"]
        assert "三只松鼠零食" not in body2["script_content"]
    finally:
        skill._script_agent = old_script_agent
        skill._quality_agent = old_quality_agent
        api_module.orchestrator.intent_agent._fast_classify = old_fast_classify
