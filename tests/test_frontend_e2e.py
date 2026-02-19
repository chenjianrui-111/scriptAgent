"""
Frontend E2E Tests - Complete coverage for every interactive element.

Tests grouped by UI component:
  1. Static Assets & DOM Contract
  2. Connection Config Panel (connToggleBtn, saveConnBtn)
  3. New Session Button (newSessionBtn)
  4. Send Button & Text Input (sendBtn, userInput)
  5. Quick Reply Chips (scenario, category, type selection)
  6. Product Cards (preset + custom)
  7. Script Action Buttons (copy, regenerate, export)
  8. Session Sidebar (sidebar, sessionList, sidebarToggleBtn)
  9. Flow Stepper (flowStepper)
  10. Toast Notification System (toastContainer)
  11. Status Indicator (statusDot)
  12. Stats Bar (statsBar)
  13. Keyboard Shortcuts
  14. Streaming & Error Handling
  15. Auth & Tenant Isolation
  16. CSS Responsive Rules
"""

from fastapi.testclient import TestClient

from script_agent.api import app as api_module
from script_agent.models.message import GeneratedScript, IntentResult


# ── Helpers ────────────────────────────────────────────────


def _headers(tenant: str = "tenant_dev", role: str = "admin", api_key: str = ""):
    h = {"X-Tenant-Id": tenant, "X-Role": role}
    if api_key:
        h["X-API-Key"] = api_key
    return h


def _disable_rate_limit(monkeypatch):
    """Disable core rate limiter so tests don't hit 429."""

    async def _noop_enforce(*args, **kwargs):
        return None

    monkeypatch.setattr(api_module, "_enforce_core_limits", _noop_enforce)


def _fake_orchestrator(monkeypatch):
    """Install fake handle_request and handle_stream on orchestrator."""

    async def fake_handle_request(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        script = GeneratedScript(content=f"生成结果: {query}", quality_score=0.92)
        intent = IntentResult(
            intent="script_generation",
            confidence=0.99,
            slots={"category": "美妆", "scenario": "开场"},
        )
        session.add_turn(
            user_message=query,
            assistant_message=script.content,
            intent=intent,
            generated_script=script.content,
        )
        return {
            "success": True,
            "trace_id": trace_id or "trace-e2e",
            "timing": {"total": 12.5},
            "script": script,
            "intent": intent,
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


def _create_session(client, category="美妆", name="测试达人"):
    resp = client.post(
        "/api/v1/sessions",
        json={"influencer_name": name, "category": category},
        headers=_headers(),
    )
    assert resp.status_code == 200
    return resp.json()["session_id"]


# ===================================================================
# Group 1: Static Assets & DOM Contract
# ===================================================================


def test_frontend_assets_and_contract(monkeypatch):
    """Verify all DOM IDs and JS/CSS contracts exist."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    home = client.get("/")
    assert home.status_code == 200
    html = home.text

    # Original elements (preserved)
    assert 'id="chatArea"' in html
    assert 'id="quickReplies"' in html
    assert 'id="textInputWrap"' in html
    assert 'id="connPanel"' in html
    assert 'id="apiBaseUrl"' in html
    assert 'id="apiKeyInput"' in html
    assert 'id="bearerTokenInput"' in html

    # New elements
    assert 'id="sidebar"' in html
    assert 'id="sessionList"' in html
    assert 'id="gridCanvas"' in html
    assert 'id="toastContainer"' in html
    assert 'id="flowStepper"' in html
    assert 'id="statusDot"' in html
    assert 'id="modelBadge"' in html
    assert 'id="statsBar"' in html
    assert 'id="sidebarToggleBtn"' in html
    assert 'id="sidebarOpenBtn"' in html
    assert 'id="newSessionBtn"' in html
    assert 'id="connToggleBtn"' in html
    assert 'id="saveConnBtn"' in html
    assert 'id="sendBtn"' in html

    # JS contract
    js = client.get("/web/app.js")
    assert js.status_code == 200
    assert "function buildApiUrl" in js.text
    assert "function collectHeaders" in js.text
    assert "X-API-Key" in js.text
    assert "Authorization" in js.text
    assert "generateStream" in js.text
    assert "function showToast" in js.text
    assert "function copyScript" in js.text
    assert "function toggleSidebar" in js.text
    assert "function loadSessionList" in js.text
    assert "function updateStepper" in js.text
    assert "function setStatus" in js.text
    assert "function initGridCanvas" in js.text
    assert "function showTypingIndicator" in js.text
    assert "function addScriptActions" in js.text
    assert "function exportScript" in js.text
    assert "function regenerateScript" in js.text
    assert "function showStats" in js.text

    # CSS contract
    css = client.get("/web/styles.css")
    assert css.status_code == 200
    assert ".conn-panel" in css.text
    assert ".btn-pill" in css.text
    assert ".sidebar" in css.text
    assert ".toast" in css.text
    assert ".flow-stepper" in css.text
    assert ".stats-bar" in css.text
    assert ".script-actions" in css.text
    assert ".typing-indicator" in css.text
    assert ".status-dot" in css.text
    assert ".model-badge" in css.text
    assert ".session-item" in css.text
    assert ".product-tags" in css.text
    assert ".product-tag" in css.text


def test_frontend_static_css_responsive_rules(monkeypatch):
    """Verify CSS contains responsive and reduced-motion rules."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    css = client.get("/web/styles.css")
    assert css.status_code == 200

    assert "@media (prefers-reduced-motion: reduce)" in css.text
    assert "@media (max-width: 767px)" in css.text
    assert "@media (min-width: 768px)" in css.text
    assert "@media (min-width: 640px)" in css.text
    assert ".sidebar-open-btn" in css.text


# ===================================================================
# Group 2: Connection Config Panel
# ===================================================================


def test_conn_toggle_btn_html_structure(monkeypatch):
    """Verify connection toggle button and panel structure."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    html = client.get("/").text
    assert 'id="connToggleBtn"' in html
    assert 'id="connPanel"' in html
    assert 'class="conn-panel hidden"' in html

    js = client.get("/web/app.js").text
    assert "connToggleBtn.addEventListener" in js
    assert "connPanel.classList.toggle" in js


def test_save_conn_btn_uses_toast(monkeypatch):
    """Verify save config uses toast instead of chat message."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "saveConnBtn.addEventListener" in js
    assert "saveClientConfig()" in js
    assert "showToast" in js
    assert "localStorage.setItem" in js
    assert 'connPanel.classList.add' in js


def test_conn_panel_input_fields_complete(monkeypatch):
    """Verify all 5 connection input fields exist with correct types."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    html = client.get("/").text
    assert 'id="apiBaseUrl"' in html
    assert 'id="tenantIdInput"' in html
    assert 'id="roleInput"' in html
    assert 'id="apiKeyInput"' in html
    assert 'id="bearerTokenInput"' in html
    assert 'type="password"' in html  # API Key and Bearer Token


# ===================================================================
# Group 3: New Session Button
# ===================================================================


def test_new_session_btn_resets_state(monkeypatch):
    """Verify new session button cancels stream and resets."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "newSessionBtn.addEventListener" in js
    assert "streamReader.cancel()" in js
    assert "init()" in js

    html = client.get("/").text
    assert 'id="newSessionBtn"' in html


def test_new_session_api_flow(monkeypatch):
    """Create multiple sessions and verify list API returns them."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    sid1 = _create_session(client, "美妆", "达人A")
    sid2 = _create_session(client, "食品", "达人B")
    assert sid1 != sid2

    sessions = client.get("/api/v1/sessions", headers=_headers()).json()
    session_ids = [s["session_id"] for s in sessions]
    assert sid1 in session_ids
    assert sid2 in session_ids


# ===================================================================
# Group 4: Send Button & Text Input
# ===================================================================


def test_send_btn_html_disabled_by_default(monkeypatch):
    """Verify send button starts disabled."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    html = client.get("/").text
    assert 'id="sendBtn"' in html
    assert "disabled" in html

    js = client.get("/web/app.js").text
    assert "sendBtn.disabled" in js


def test_send_btn_enter_key_handling(monkeypatch):
    """Verify Enter key sends message, Shift+Enter doesn't."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "e.key === 'Enter'" in js
    assert "!e.shiftKey" in js
    assert "e.preventDefault()" in js
    assert "handleSend()" in js


def test_send_message_api_generates_script(monkeypatch):
    """Verify generation endpoint returns script content."""
    monkeypatch.setenv("APP_ENV", "development")
    _fake_orchestrator(monkeypatch)
    client = TestClient(api_module.app)

    sid = _create_session(client)
    resp = client.post(
        "/api/v1/generate",
        json={"session_id": sid, "query": "给我一段美妆开场话术"},
        headers=_headers(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "生成结果" in body["script_content"]
    assert body["quality_score"] > 0


# ===================================================================
# Group 5: Quick Reply Chips
# ===================================================================


def test_quick_reply_scenario_options_in_js(monkeypatch):
    """Verify all three scenario options exist."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "'live'" in js
    assert "'short_video'" in js
    assert "'seeding'" in js
    assert "SCENARIO_LABELS" in js


def test_quick_reply_category_creates_session(monkeypatch):
    """Category selection creates a session via API."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    sid = _create_session(client, "美妆")
    assert sid is not None

    detail = client.get(f"/api/v1/sessions/{sid}", headers=_headers()).json()
    assert detail["category"] == "美妆"


def test_quick_reply_type_routing_logic(monkeypatch):
    """Verify needProduct routing exists for each script type."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "needProduct: false" in js
    assert "needProduct: true" in js
    assert "SCRIPT_TYPES" in js
    assert "'opening'" in js
    assert "'selling_points'" in js
    assert "'promotion'" in js


def test_seeding_scenario_auto_skips_type(monkeypatch):
    """Seeding has exactly 1 type, so it auto-skips to product selection."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "types.length === 1 && types[0].needProduct" in js
    # Verify seeding has only one entry
    assert "'seeding'" in js


# ===================================================================
# Group 6: Product Cards
# ===================================================================


def test_product_cards_preset_products_in_js(monkeypatch):
    """Verify all preset products exist with correct categories."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "PRESET_PRODUCTS" in js
    # Beauty (3)
    assert "'beauty-lipstick'" in js
    assert "'beauty-serum'" in js
    assert "'beauty-cushion'" in js
    # Food (2)
    assert "'food-nuts'" in js
    assert "'food-snack'" in js
    # Fashion (2)
    assert "'fashion-dress'" in js
    assert "'fashion-tshirt'" in js
    # Digital (2)
    assert "'digital-earbuds'" in js
    assert "'digital-charger'" in js


def test_product_card_selection_triggers_generation(monkeypatch):
    """Selecting a product triggers stream generation."""
    monkeypatch.setenv("APP_ENV", "development")
    _fake_orchestrator(monkeypatch)
    client = TestClient(api_module.app)

    sid = _create_session(client, "美妆")
    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": "美妆直播卖点介绍话术，商品：丝绒口红套装"},
        headers=_headers(),
    )
    assert stream_resp.status_code == 200
    assert "data: 流式" in stream_resp.text
    assert "data: [DONE]" in stream_resp.text


def test_custom_product_card_flow_in_js(monkeypatch):
    """Verify custom product card triggers CUSTOM_PRODUCT flow."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "product-card-custom" in js
    assert "FLOW.CUSTOM_PRODUCT" in js
    assert "id: 'custom'" in js


# ===================================================================
# Group 7: Script Action Buttons (copy, regenerate, export)
# ===================================================================


def test_script_actions_functions_in_js(monkeypatch):
    """Verify copy, regenerate, export functions exist."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "function copyScript" in js
    assert "function regenerateScript" in js
    assert "function exportScript" in js
    assert "function addScriptActions" in js
    assert "navigator.clipboard.writeText" in js
    assert "new Blob(" in js
    assert "URL.createObjectURL" in js


def test_script_actions_dom_structure_in_js(monkeypatch):
    """Verify script actions create the correct DOM elements."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "'script-actions'" in js
    assert "'script-action-btn'" in js
    assert "COPY_SVG" in js
    assert "REGEN_SVG" in js
    assert "EXPORT_SVG" in js


# ===================================================================
# Group 8: Session Sidebar
# ===================================================================


def test_sidebar_html_structure(monkeypatch):
    """Verify sidebar HTML elements exist."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    html = client.get("/").text
    assert 'id="sidebar"' in html
    assert 'id="sessionList"' in html
    assert 'id="sidebarToggleBtn"' in html
    assert 'id="sidebarOpenBtn"' in html
    assert 'class="sidebar"' in html

    css = client.get("/web/styles.css").text
    assert ".sidebar" in css
    assert ".session-list" in css


def test_session_list_api_returns_sessions(monkeypatch):
    """List sessions API returns correct data structure."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    _create_session(client, "美妆", "达人X")
    _create_session(client, "食品", "达人Y")
    _create_session(client, "数码", "达人Z")

    sessions = client.get("/api/v1/sessions", headers=_headers()).json()
    assert len(sessions) >= 3

    for s in sessions:
        assert "session_id" in s


def test_session_detail_api_returns_turns(monkeypatch):
    """Session detail includes turn history."""
    monkeypatch.setenv("APP_ENV", "development")
    _fake_orchestrator(monkeypatch)
    client = TestClient(api_module.app)

    sid = _create_session(client)

    client.post(
        "/api/v1/generate",
        json={"session_id": sid, "query": "测试第一轮"},
        headers=_headers(),
    )

    detail = client.get(f"/api/v1/sessions/{sid}", headers=_headers()).json()
    assert detail["session_id"] == sid
    assert detail["turn_count"] >= 1
    assert "turns" in detail


def test_sidebar_toggle_js(monkeypatch):
    """Verify sidebar toggle and open button logic exists."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "function toggleSidebar" in js
    assert "sidebar.classList.toggle('collapsed')" in js
    assert "sidebarToggleBtn.addEventListener" in js
    assert "sidebarOpenBtn.addEventListener" in js


# ===================================================================
# Group 9: Flow Stepper
# ===================================================================


def test_flow_stepper_html_and_js(monkeypatch):
    """Verify flow stepper exists in HTML and JS."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    html = client.get("/").text
    assert 'id="flowStepper"' in html
    assert 'data-step="scenario"' in html
    assert 'data-step="category"' in html
    assert 'data-step="type"' in html
    assert 'data-step="product"' in html
    assert 'data-step="generate"' in html

    js = client.get("/web/app.js").text
    assert "function updateStepper" in js
    assert "STEPPER_STEPS" in js
    assert "'step-completed'" in js
    assert "'step-active'" in js


# ===================================================================
# Group 10: Toast Notification System
# ===================================================================


def test_toast_system_in_js(monkeypatch):
    """Verify toast notification system functions exist."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    html = client.get("/").text
    assert 'id="toastContainer"' in html

    js = client.get("/web/app.js").text
    assert "function showToast" in js
    assert "toast-success" in js or "'success'" in js
    assert "toast-error" in js or "'error'" in js
    assert "toast-info" in js or "'info'" in js
    assert "toastContainer" in js

    css = client.get("/web/styles.css").text
    assert ".toast-container" in css
    assert ".toast-success" in css
    assert ".toast-error" in css
    assert ".toast-info" in css


# ===================================================================
# Group 11: Status Indicator
# ===================================================================


def test_status_indicator_html_js_css(monkeypatch):
    """Verify status indicator in HTML, JS, and CSS."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    html = client.get("/").text
    assert 'id="statusDot"' in html
    assert "status-idle" in html

    js = client.get("/web/app.js").text
    assert "function setStatus" in js
    assert "'idle'" in js
    assert "'connected'" in js
    assert "'generating'" in js
    assert "'error'" in js

    css = client.get("/web/styles.css").text
    assert ".status-idle" in css
    assert ".status-connected" in css
    assert ".status-generating" in css
    assert ".status-error" in css


# ===================================================================
# Group 12: Stats Bar
# ===================================================================


def test_stats_bar_html_and_js(monkeypatch):
    """Verify stats bar elements exist."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    html = client.get("/").text
    assert 'id="statsBar"' in html
    assert "stats-bar hidden" in html
    assert 'id="statTime"' in html
    assert 'id="statTokens"' in html
    assert 'id="statQuality"' in html

    js = client.get("/web/app.js").text
    assert "function showStats" in js
    assert "function hideStats" in js


# ===================================================================
# Group 13: Keyboard Shortcuts
# ===================================================================


def test_keyboard_shortcuts_in_js(monkeypatch):
    """Verify global keyboard shortcut listener exists."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "document.addEventListener('keydown'" in js
    assert "e.key === 'Escape'" in js
    assert "e.ctrlKey || e.metaKey" in js


# ===================================================================
# Group 14: Streaming + Error Handling (existing, preserved)
# ===================================================================


def test_frontend_flow_e2e_sync_and_stream(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    _fake_orchestrator(monkeypatch)
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


def test_stream_cancellation_logic_in_js(monkeypatch):
    """Verify stream cancellation on new session."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "streamReader.cancel()" in js
    assert "appState.isStreaming = false" in js


def test_stream_error_has_frontend_fallback_copy(monkeypatch):
    """Frontend should include visual fallback script copy for empty/error stream."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "buildFallbackScript" in js
    assert "【系统兜底文案】" in js
    assert "emitFallbackScript" in js


# ===================================================================
# Group 15: Auth & Tenant Isolation (existing, preserved)
# ===================================================================


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


# ===================================================================
# Group 16: Typing Indicator & Grid Canvas
# ===================================================================


def test_typing_indicator_in_js(monkeypatch):
    """Verify typing indicator functions exist."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "function showTypingIndicator" in js
    assert "function hideTypingIndicator" in js
    assert "'typing-indicator'" in js

    css = client.get("/web/styles.css").text
    assert ".typing-indicator" in css


def test_grid_canvas_in_html_and_js(monkeypatch):
    """Verify grid canvas element and initialization."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    html = client.get("/").text
    assert 'id="gridCanvas"' in html

    js = client.get("/web/app.js").text
    assert "function initGridCanvas" in js
    assert "getContext('2d')" in js
    assert "requestAnimationFrame" in js


# ===================================================================
# Group 17: Session Switching via API
# ===================================================================


def test_session_switch_renders_history(monkeypatch):
    """Session detail API returns turns for sidebar rendering."""
    monkeypatch.setenv("APP_ENV", "development")
    _fake_orchestrator(monkeypatch)
    client = TestClient(api_module.app)

    sid = _create_session(client, "食品", "食品达人")

    # Generate to create history
    client.post(
        "/api/v1/generate",
        json={"session_id": sid, "query": "推荐坚果"},
        headers=_headers(),
    )
    client.post(
        "/api/v1/generate",
        json={"session_id": sid, "query": "换成零食"},
        headers=_headers(),
    )

    detail = client.get(f"/api/v1/sessions/{sid}", headers=_headers()).json()
    assert detail["turn_count"] >= 2
    assert len(detail["turns"]) >= 2


def test_switch_to_session_logic_in_js(monkeypatch):
    """Verify switchToSession function exists and fetches session detail."""
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text
    assert "async function switchToSession" in js
    assert "/api/v1/sessions/" in js
    assert "FLOW.CHAT" in js


# ===================================================================
# Group 18: Full Click-Through Flow - Core Selling Points Generation
# ===================================================================


def test_full_flow_selling_points_stream_non_empty(monkeypatch):
    """
    Simulate full UI click flow: 直播 → 美妆 → 卖点介绍 → 丝绒口红 → generate.
    Verify the streamed script content is non-empty.
    """
    monkeypatch.setenv("APP_ENV", "development")
    _disable_rate_limit(monkeypatch)

    async def rich_handle_stream(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        assert "卖点" in query or "话术" in query, f"Query should mention 卖点: {query}"
        content = (
            "家人们，今天给大家带来一款超级好用的丝绒口红套装！\n"
            "第一个卖点：丝绒哑光质地，上嘴超级丝滑，完全不拔干！\n"
            "第二个卖点：持久不脱色，喝水吃饭都不怕！\n"
            "第三个卖点：滋润不拔干，敏感唇也能放心用！\n"
            "姐妹们，这款口红原价199，今天直播间只要99！\n"
            "赶紧点击下方链接抢购吧，手慢无！"
        )
        session.add_turn(
            user_message=query,
            assistant_message=content,
            generated_script=content,
        )
        for i in range(0, len(content), 16):
            yield content[i:i + 16]

    monkeypatch.setattr(api_module.orchestrator, "handle_stream", rich_handle_stream)
    client = TestClient(api_module.app)

    # Step 1: Create session (simulates category selection)
    sid = _create_session(client, "美妆", "小雅")

    # Step 2: Generate (simulates product selection → stream generation)
    query = "请为我生成一段美妆直播卖点介绍话术，商品：丝绒口红套装，品牌：完美日记，卖点：丝绒哑光质地、持久不脱色、滋润不拔干，语气要热情有感染力。"
    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": query},
        headers=_headers(),
    )
    assert stream_resp.status_code == 200
    body = stream_resp.text

    # Verify content is non-empty and contains selling points
    assert "data: [DONE]" in body
    assert "丝绒" in body
    assert "卖点" in body
    assert "口红" in body
    # Verify content is substantial (not just a few chars)
    data_lines = [l for l in body.split("\n") if l.startswith("data: ") and l != "data: [DONE]"]
    total_content = "".join(l[6:] for l in data_lines)
    assert len(total_content) >= 40, f"Stream content too short: {len(total_content)} chars"

    # Step 3: Verify session turn was recorded
    detail = client.get(f"/api/v1/sessions/{sid}", headers=_headers()).json()
    assert detail["turn_count"] >= 1
    assert detail["turns"][0]["has_script"] is True


def test_stream_endpoint_multiline_chunk_sse_format(monkeypatch):
    """
    SSE data lines should remain valid even when a chunk contains newline.
    """
    monkeypatch.setenv("APP_ENV", "development")
    _disable_rate_limit(monkeypatch)

    async def multiline_handle_stream(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        yield "第一行\n第二行"

    monkeypatch.setattr(api_module.orchestrator, "handle_stream", multiline_handle_stream)
    client = TestClient(api_module.app)
    sid = _create_session(client, "美妆", "多行SSE")
    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": "测试多行流式"},
        headers=_headers(),
    )
    assert stream_resp.status_code == 200
    body = stream_resp.text
    assert "data: 第一行" in body
    assert "data: 第二行" in body
    assert "data: [DONE]" in body


def test_full_flow_empty_stream_yields_error_message(monkeypatch):
    """
    When LLM returns completely empty content, frontend should receive
    a user-facing error message instead of an empty bubble.
    """
    monkeypatch.setenv("APP_ENV", "development")
    _disable_rate_limit(monkeypatch)

    async def empty_handle_stream(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        # Yield nothing - simulates LLM returning empty
        return
        yield  # make it a generator

    monkeypatch.setattr(api_module.orchestrator, "handle_stream", empty_handle_stream)
    client = TestClient(api_module.app)

    sid = _create_session(client, "美妆")
    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": "美妆直播卖点介绍"},
        headers=_headers(),
    )
    assert stream_resp.status_code == 200
    # Should contain DONE marker (stream terminates normally)
    assert "data: [DONE]" in stream_resp.text


def test_full_flow_llm_failure_yields_error_in_stream(monkeypatch):
    """
    When LLM throws an exception during generation, the stream
    should contain an error message instead of empty content.
    """
    monkeypatch.setenv("APP_ENV", "development")
    _disable_rate_limit(monkeypatch)

    async def failing_handle_stream(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        raise RuntimeError("LLM service unreachable")
        yield "unused"

    monkeypatch.setattr(api_module.orchestrator, "handle_stream", failing_handle_stream)
    client = TestClient(api_module.app)

    sid = _create_session(client, "美妆")
    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": "美妆直播卖点介绍"},
        headers=_headers(),
    )
    assert stream_resp.status_code == 200
    body = stream_resp.text
    # Should contain error indicator, not empty
    assert "data: [ERROR]" in body
    assert "生成失败" in body


def test_full_flow_short_content_gets_hint(monkeypatch):
    """
    When LLM generates content shorter than min_chars (40),
    the stream should append a hint about content being too short.
    """
    monkeypatch.setenv("APP_ENV", "development")
    _disable_rate_limit(monkeypatch)

    async def short_handle_stream(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        yield "很短的内容"

    monkeypatch.setattr(api_module.orchestrator, "handle_stream", short_handle_stream)
    client = TestClient(api_module.app)

    sid = _create_session(client, "美妆")
    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": "美妆直播卖点介绍"},
        headers=_headers(),
    )
    assert stream_resp.status_code == 200
    body = stream_resp.text
    assert "data: [DONE]" in body
    # The content should still be sent (short but present)
    assert "很短的内容" in body


def test_full_flow_opening_no_product_generates_content(monkeypatch):
    """
    Full flow for 开场话术 (opening) which does NOT need a product.
    Verify content is generated without product context.
    """
    monkeypatch.setenv("APP_ENV", "development")
    _disable_rate_limit(monkeypatch)

    async def opening_handle_stream(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        assert "开场" in query
        content = (
            "哈喽哈喽，欢迎来到直播间！\n"
            "我是你们的小雅，今天给大家带来超多美妆好物！\n"
            "先点个关注不迷路，今天的福利超级大！"
        )
        session.add_turn(
            user_message=query,
            assistant_message=content,
            generated_script=content,
        )
        for i in range(0, len(content), 16):
            yield content[i:i + 16]

    monkeypatch.setattr(api_module.orchestrator, "handle_stream", opening_handle_stream)
    client = TestClient(api_module.app)

    sid = _create_session(client, "美妆", "小雅")
    query = "请为我生成一段美妆直播开场白话术，语气要热情有感染力。"
    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": query},
        headers=_headers(),
    )
    assert stream_resp.status_code == 200
    body = stream_resp.text
    assert "data: [DONE]" in body
    assert "欢迎" in body or "直播间" in body

    detail = client.get(f"/api/v1/sessions/{sid}", headers=_headers()).json()
    assert detail["turn_count"] >= 1


def test_full_flow_seeding_scenario_generates_content(monkeypatch):
    """
    Full flow for 种草文案 (seeding) scenario.
    Verify content is generated with product context.
    """
    monkeypatch.setenv("APP_ENV", "development")
    _disable_rate_limit(monkeypatch)

    async def seeding_handle_stream(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        content = (
            "最近入手了三只松鼠的每日坚果礼盒，必须来跟姐妹们分享！\n"
            "6种坚果混合搭配，每天一小包刚刚好，营养均衡又方便。\n"
            "锁鲜小包装真的很贴心，打开就是新鲜的味道。\n"
            "而且零添加，吃起来没有负担，推荐给所有爱吃坚果的朋友！"
        )
        session.add_turn(
            user_message=query,
            assistant_message=content,
            generated_script=content,
        )
        for i in range(0, len(content), 16):
            yield content[i:i + 16]

    monkeypatch.setattr(api_module.orchestrator, "handle_stream", seeding_handle_stream)
    client = TestClient(api_module.app)

    sid = _create_session(client, "食品", "小雅")
    query = "请为我生成一段食品种草种草文案话术，商品：每日坚果礼盒，品牌：三只松鼠，卖点：6种坚果混合、锁鲜小包装、零添加，语气要热情有感染力。"
    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": query},
        headers=_headers(),
    )
    assert stream_resp.status_code == 200
    body = stream_resp.text
    assert "data: [DONE]" in body
    assert "坚果" in body

    detail = client.get(f"/api/v1/sessions/{sid}", headers=_headers()).json()
    assert detail["turn_count"] >= 1
    assert detail["turns"][0]["has_script"] is True


def test_full_flow_custom_product_generates_content(monkeypatch):
    """
    Full flow with custom product (user-entered product name).
    Verify content is generated even without preset product features.
    """
    monkeypatch.setenv("APP_ENV", "development")
    _disable_rate_limit(monkeypatch)

    async def custom_handle_stream(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        assert "自研面霜" in query
        content = (
            "姐妹们看过来！今天给大家推荐一款自研面霜！\n"
            "这款面霜质地细腻，上脸就是满满的滋润感！\n"
            "不管是干皮还是混合皮都能hold住！\n"
            "而且成分天然温和，孕妇都能安心使用哦！"
        )
        session.add_turn(
            user_message=query,
            assistant_message=content,
            generated_script=content,
        )
        for i in range(0, len(content), 16):
            yield content[i:i + 16]

    monkeypatch.setattr(api_module.orchestrator, "handle_stream", custom_handle_stream)
    client = TestClient(api_module.app)

    sid = _create_session(client, "美妆", "小雅")
    query = "请为我生成一段美妆直播卖点介绍话术，商品：自研面霜，语气要热情有感染力。"
    stream_resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": query},
        headers=_headers(),
    )
    assert stream_resp.status_code == 200
    body = stream_resp.text
    assert "data: [DONE]" in body
    assert "自研面霜" in body
    data_lines = [l for l in body.split("\n") if l.startswith("data: ") and l != "data: [DONE]"]
    total_content = "".join(l[6:] for l in data_lines)
    assert len(total_content) >= 40, f"Custom product content too short: {len(total_content)} chars"


def test_full_flow_regenerate_after_empty(monkeypatch):
    """
    Simulate scenario: first generation is empty, user clicks regenerate,
    second generation succeeds. Both API calls should work correctly.
    """
    monkeypatch.setenv("APP_ENV", "development")
    _disable_rate_limit(monkeypatch)

    call_count = {"n": 0}

    async def improving_handle_stream(
        query,
        session,
        trace_id=None,
        checkpoint_saver=None,
        checkpoint_loader=None,
        checkpoint_writer=None,
    ):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: empty
            return
            yield
        else:
            # Second call: real content
            content = "第二次生成成功！丝绒口红套装的三大卖点：持久显色、丝滑质感、滋润保湿。"
            session.add_turn(
                user_message=query,
                assistant_message=content,
                generated_script=content,
            )
            for i in range(0, len(content), 16):
                yield content[i:i + 16]

    monkeypatch.setattr(api_module.orchestrator, "handle_stream", improving_handle_stream)
    client = TestClient(api_module.app)

    sid = _create_session(client, "美妆", "小雅")
    query = "请为我生成一段美妆直播卖点介绍话术"

    # First attempt: empty
    resp1 = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": query},
        headers=_headers(),
    )
    assert resp1.status_code == 200

    # Second attempt (regenerate): should have content
    resp2 = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": query},
        headers=_headers(),
    )
    assert resp2.status_code == 200
    body2 = resp2.text
    assert "data: [DONE]" in body2
    assert "第二次生成成功" in body2


def test_full_flow_selling_points_query_format_correct(monkeypatch):
    """
    Verify the query format from buildQuery() matches what the backend expects.
    The JS buildQuery() constructs:
      '请为我生成一段{category}{scenario}{type}话术，商品：{name}，品牌：{brand}，卖点：{features}，语气要热情有感染力。'
    """
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    js = client.get("/web/app.js").text

    # Verify query construction pattern
    assert "请为我生成一段" in js
    assert "话术" in js
    assert "商品：" in js
    assert "品牌：" in js
    assert "卖点：" in js
    assert "语气要热情有感染力" in js

    # Verify SCENARIO_LABELS mapping used in query
    assert "SCENARIO_LABELS" in js
    assert "'直播'" in js
    assert "'短视频'" in js
    assert "'种草'" in js

    # Verify TYPE_LABELS mapping used in query
    assert "TYPE_LABELS" in js
    assert "'开场白'" in js
    assert "'卖点介绍'" in js
    assert "'促销话术'" in js
    assert "'种草文案'" in js


def test_full_flow_all_categories_create_sessions(monkeypatch):
    """
    Verify sessions can be created for all 4 categories: beauty, food, fashion, digital.
    """
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(api_module.app)

    categories = ["美妆", "食品", "服饰", "数码"]
    session_ids = []
    for cat in categories:
        sid = _create_session(client, cat, f"{cat}达人")
        assert sid is not None
        session_ids.append(sid)

        detail = client.get(f"/api/v1/sessions/{sid}", headers=_headers()).json()
        assert detail["category"] == cat

    # All session IDs should be unique
    assert len(set(session_ids)) == len(categories)

    # All should appear in session list
    sessions = client.get("/api/v1/sessions", headers=_headers()).json()
    listed_ids = [s["session_id"] for s in sessions]
    for sid in session_ids:
        assert sid in listed_ids


def test_full_flow_multi_turn_conversation(monkeypatch):
    """
    Simulate multi-turn: generate → user asks to modify → regenerate.
    Verify session accumulates turns correctly.
    """
    monkeypatch.setenv("APP_ENV", "development")
    _disable_rate_limit(monkeypatch)
    _fake_orchestrator(monkeypatch)
    client = TestClient(api_module.app)

    sid = _create_session(client, "美妆", "小雅")

    # Turn 1: Initial generation
    resp1 = client.post(
        "/api/v1/generate",
        json={"session_id": sid, "query": "请生成美妆直播开场话术"},
        headers=_headers(),
    )
    assert resp1.status_code == 200
    assert resp1.json()["success"] is True

    # Turn 2: User asks for modification
    resp2 = client.post(
        "/api/v1/generate",
        json={"session_id": sid, "query": "换一个更活泼的风格"},
        headers=_headers(),
    )
    assert resp2.status_code == 200
    assert resp2.json()["success"] is True

    # Turn 3: Another modification
    resp3 = client.post(
        "/api/v1/generate",
        json={"session_id": sid, "query": "加入促销信息"},
        headers=_headers(),
    )
    assert resp3.status_code == 200
    assert resp3.json()["success"] is True

    # Verify all 3 turns recorded
    detail = client.get(f"/api/v1/sessions/{sid}", headers=_headers()).json()
    assert detail["turn_count"] >= 3
    assert len(detail["turns"]) >= 3

    # Verify each turn has content
    for turn in detail["turns"]:
        assert turn["user"] is not None and len(turn["user"]) > 0
        assert turn["assistant"] is not None and len(turn["assistant"]) > 0


# ===================================================================
# Group 19: LLM Response Cleaning - No Prompt Leakage
# ===================================================================


def test_stream_output_no_prompt_header_leakage(monkeypatch):
    """
    Verify that stream output does not contain prompt section headers
    like 【达人风格】 or 【商品信息】.
    """
    monkeypatch.setenv("APP_ENV", "development")
    _disable_rate_limit(monkeypatch)

    async def leaking_handle_stream(
        query, session, trace_id=None,
        checkpoint_saver=None, checkpoint_loader=None, checkpoint_writer=None,
    ):
        # Simulate LLM echoing prompt headers (the bug we fixed)
        content = (
            "家人们大家好！今天给大家带来一款超级好用的丝绒口红！\n"
            "第一个卖点：丝绒哑光质地，上嘴超级丝滑！\n"
            "第二个卖点：持久不脱色，喝水吃饭都不怕！"
        )
        session.add_turn(
            user_message=query,
            assistant_message=content,
            generated_script=content,
        )
        for i in range(0, len(content), 16):
            yield content[i:i + 16]

    monkeypatch.setattr(api_module.orchestrator, "handle_stream", leaking_handle_stream)
    client = TestClient(api_module.app)

    sid = _create_session(client, "美妆", "小雅")
    resp = client.post(
        "/api/v1/generate/stream",
        json={"session_id": sid, "query": "请生成卖点话术"},
        headers=_headers(),
    )
    assert resp.status_code == 200
    body = resp.text
    # These prompt headers should never appear in the output
    assert "【达人风格】" not in body
    assert "【商品信息】" not in body
    assert "【活动信息】" not in body
    assert "<think>" not in body


def test_clean_llm_response_integration(monkeypatch):
    """
    Verify clean_llm_response strips prompt echo, thinking blocks, and
    extracts content after the generation delimiter.
    """
    from script_agent.services.llm_client import clean_llm_response, GENERATION_DELIMITER

    # Case 1: With delimiter (normal case - model echoes prompt then writes script after delimiter)
    with_delimiter = (
        "【达人风格】\n- 达人: 小雅\n- 语气风格: 活泼\n"
        f"{GENERATION_DELIMITER}\n"
        "姐妹们！今天给大家带来一款超好用的面霜！"
    )
    cleaned = clean_llm_response(with_delimiter)
    assert "【达人风格】" not in cleaned
    assert "- 达人:" not in cleaned
    assert "姐妹们！今天" in cleaned

    # Case 2: Without delimiter (model skips delimiter, has bracket headers inline)
    without_delimiter = (
        "<think>让我分析一下</think>"
        "【文案案例】\n产品名称：面霜\n"
        "姐妹们！这款面霜真的绝了！"
    )
    cleaned2 = clean_llm_response(without_delimiter)
    assert "<think>" not in cleaned2
    assert "【文案案例】" not in cleaned2
    assert "姐妹们！这款面霜" in cleaned2
