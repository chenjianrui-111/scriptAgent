import asyncio

import pytest

from script_agent.services.llm_client import (
    CircuitBreaker,
    LLMBackend,
    LLMCallError,
    LLMServiceClient,
    clean_llm_response,
)


class DummyBackend(LLMBackend):
    def __init__(self, sync_plan=None, stream_plan=None):
        super().__init__()
        self.sync_plan = list(sync_plan or [])
        self.stream_plan = list(stream_plan or [])
        self.sync_calls = 0
        self.stream_calls = 0
        self.sync_kwargs = []

    async def _resolve_action(self, action):
        if callable(action):
            action = action()
        if asyncio.iscoroutine(action):
            action = await action
        if isinstance(action, Exception):
            raise action
        return action

    async def generate(self, prompt: str, category: str = "通用",
                       stream: bool = True, **kwargs):
        self.stream_calls += 1
        action = self.stream_plan.pop(0) if self.stream_plan else [""]
        action = await self._resolve_action(action)
        if isinstance(action, str):
            yield action
            return
        for token in action:
            yield token

    async def generate_sync(self, prompt: str, category: str = "通用",
                            max_tokens: int = 1024, **kwargs) -> str:
        self.sync_calls += 1
        self.sync_kwargs.append(kwargs)
        action = self.sync_plan.pop(0) if self.sync_plan else ""
        action = await self._resolve_action(action)
        return str(action)

    async def health_check(self):
        return {"status": "healthy"}


def _build_client(primary: LLMBackend, fallback: LLMBackend | None = None) -> LLMServiceClient:
    client = LLMServiceClient(env="development")
    client.backend = primary
    client._fallback_backend = fallback
    client._retry_max_attempts = 2
    client._retry_base_delay_seconds = 0.0
    client._retry_max_delay_seconds = 0.0
    client._retry_jitter_seconds = 0.0
    client._primary_breaker = CircuitBreaker(
        "primary-test",
        enabled=False,
        failure_threshold=2,
        recovery_seconds=30,
        half_open_max_calls=1,
    )
    client._fallback_breaker = CircuitBreaker(
        "fallback-test",
        enabled=False,
        failure_threshold=2,
        recovery_seconds=30,
        half_open_max_calls=1,
    )
    return client


@pytest.mark.asyncio
async def test_retry_on_retryable_error_then_success():
    primary = DummyBackend(sync_plan=[
        LLMCallError(
            "temporary error",
            retryable=True,
            fallback_eligible=True,
        ),
        "ok-after-retry",
    ])
    client = _build_client(primary)

    result = await client.generate_sync("prompt", category="通用")

    assert result == "ok-after-retry"
    assert primary.sync_calls == 2


@pytest.mark.asyncio
async def test_fallback_layer_when_primary_failed():
    primary = DummyBackend(sync_plan=[
        LLMCallError(
            "model missing",
            retryable=False,
            fallback_eligible=True,
            status_code=404,
        )
    ])
    fallback = DummyBackend(sync_plan=["from-fallback"])
    client = _build_client(primary, fallback)

    result = await client.generate_sync("prompt", category="通用")

    assert result == "from-fallback"
    assert primary.sync_calls == 1
    assert fallback.sync_calls == 1


@pytest.mark.asyncio
async def test_prefer_fallback_routes_to_backup_model():
    primary = DummyBackend(sync_plan=["from-primary"])
    fallback = DummyBackend(sync_plan=["from-fallback-direct"])
    client = _build_client(primary, fallback)

    result = await client.generate_sync(
        "prompt",
        category="美妆",
        prefer_fallback=True,
    )

    assert result == "from-fallback-direct"
    assert primary.sync_calls == 0
    assert fallback.sync_calls == 1


@pytest.mark.asyncio
async def test_sync_idempotency_inflight_dedup():
    async def slow_result():
        await asyncio.sleep(0.05)
        return "slow-ok"

    primary = DummyBackend(sync_plan=[slow_result])
    client = _build_client(primary)
    client._retry_max_attempts = 1
    client._sync_inflight_enabled = True

    r1, r2 = await asyncio.gather(
        client.generate_sync("same prompt", category="通用", idempotency_key="idem-1"),
        client.generate_sync("same prompt", category="通用", idempotency_key="idem-1"),
    )

    assert r1 == "slow-ok"
    assert r2 == "slow-ok"
    assert primary.sync_calls == 1


@pytest.mark.asyncio
async def test_circuit_breaker_open_skips_primary_and_use_fallback():
    primary = DummyBackend(sync_plan=[
        LLMCallError(
            "backend down",
            retryable=False,
            fallback_eligible=True,
        ),
        "should-not-be-called",
    ])
    fallback = DummyBackend(sync_plan=["fb-1", "fb-2"])
    client = _build_client(primary, fallback)
    client._retry_max_attempts = 1
    client._primary_breaker = CircuitBreaker(
        "primary-test",
        enabled=True,
        failure_threshold=1,
        recovery_seconds=60,
        half_open_max_calls=1,
    )

    first = await client.generate_sync("prompt", category="通用")
    second = await client.generate_sync("prompt2", category="通用")

    assert first == "fb-1"
    assert second == "fb-2"
    assert primary.sync_calls == 1
    assert fallback.sync_calls == 2


def test_clean_llm_response_strips_prompt_echo_markdown_blocks():
    raw = (
        "### 商品名：低卡魔芋爽\n"
        "- **核心卖点**：低卡零食、多种口味\n"
        "本轮要求：语气保持一致\n"
        "---话术正文---\n"
        "家人们，今天上新的是卫龙辣条，香辣带劲，越嚼越香！"
    )
    cleaned = clean_llm_response(raw)
    assert "商品名" not in cleaned
    assert "本轮要求" not in cleaned
    assert "卫龙辣条" in cleaned


def test_clean_llm_response_strips_inline_prompt_echo_fragments():
    raw = (
        "---话术正文---\n"
        "姐妹们，今天主推卫龙辣条，香辣过瘾超上头！"
        "（语气保持一致，但信息表达要有新增，不要整段复述）\n"
        "现在下单还有直播间专属加赠。"
    )
    cleaned = clean_llm_response(raw)
    assert "语气保持一致" not in cleaned
    assert "不要整段复述" not in cleaned
    assert "卫龙辣条" in cleaned
