import os
import uuid

import pytest
from fastapi.testclient import TestClient

from script_agent.api import app as api_module
from script_agent.config.settings import settings


def _enabled() -> bool:
    return os.getenv("RUN_REAL_LLM_E2E", "").strip().lower() in {"1", "true", "yes"}


def _headers() -> dict:
    headers = {
        "X-Tenant-Id": os.getenv("REAL_E2E_TENANT_ID", "tenant_dev"),
        "X-Role": os.getenv("REAL_E2E_ROLE", "admin"),
    }
    api_key = os.getenv("REAL_E2E_API_KEY", "").strip()
    bearer = os.getenv("REAL_E2E_BEARER_TOKEN", "").strip()
    if api_key:
        headers["X-API-Key"] = api_key
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return headers


@pytest.mark.skipif(not _enabled(), reason="set RUN_REAL_LLM_E2E=1 to run real LLM API e2e")
def test_api_e2e_product_switch_real_llm_output():
    """
    真实链路 API E2E（不 mock orchestrator / skill / LLM 输出）:
      /sessions -> /generate(旧商品) -> /generate(换品)
    断言:
      - 第二轮文案必须包含新商品名
      - 第二轮文案不应包含旧商品名
    """
    if settings.llm.primary_backend == "zhipu" and not settings.llm.zhipu_api_key:
        pytest.skip("ZHIPU_API_KEY missing for real llm e2e")

    client = TestClient(api_module.app)
    headers = _headers()

    create = client.post(
        "/api/v1/sessions",
        json={
            "influencer_name": f"真实链路验收达人-{uuid.uuid4().hex[:6]}",
            "category": "食品",
        },
        headers=headers,
    )
    assert create.status_code == 200, create.text
    session_id = create.json()["session_id"]

    first = client.post(
        "/api/v1/generate",
        json={
            "session_id": session_id,
            "query": "请生成一段食品直播卖点介绍话术，商品是三只松鼠零食，语气活泼。",
        },
        headers=headers,
    )
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body.get("success") is True, first_body
    old_script = str(first_body.get("script_content", ""))
    assert "三只松鼠零食" in old_script

    second = client.post(
        "/api/v1/generate",
        json={
            "session_id": session_id,
            "query": "不要三只松鼠零食了，换成卫龙辣条，继续直播卖点介绍，保持活泼风格。",
        },
        headers=headers,
    )
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert second_body.get("success") is True, second_body
    new_script = str(second_body.get("script_content", ""))
    assert "卫龙辣条" in new_script
    assert "三只松鼠零食" not in new_script
