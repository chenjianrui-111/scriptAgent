"""
统一LLM调用接口 - 屏蔽vLLM和Ollama的差异

开发环境 → Ollama (本地)
测试环境 → Ollama (服务器)
生产环境 → vLLM (K8s集群, 支持LoRA动态切换)

aiohttp.ClientSession 在后端生命周期内复用, 避免每次请求创建新连接池。
"""

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import aiohttp

from script_agent.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM response cleaning: strip prompt echo, thinking blocks, metadata lines
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Delimiter injected at end of prompt; everything before it is prompt echo
GENERATION_DELIMITER = "---话术正文---"

_BRACKET_HEADER_RE = re.compile(r"【[^】]{1,20}】")

_PROMPT_META_PREFIXES = (
    "- 达人:", "- 达人：", "- 语气风格:", "- 语气风格：",
    "- 常用口头禅:", "- 常用口头禅：", "- 正式度:", "- 正式度：",
    "- 目标受众:", "- 目标受众：", "- 商品名:", "- 商品名：",
    "- 品牌:", "- 品牌：", "- 价格带:", "- 价格带：",
    "- 核心特征:", "- 核心特征：", "- 主卖点:", "- 主卖点：",
    "- 合规提醒:", "- 合规提醒：", "- 适当加入",
    "角色说明：", "角色说明:", "产品名称：", "产品名称:", "卖点：", "卖点:",
)


def clean_llm_response(text: str) -> str:
    """Strip thinking blocks, prompt echo, and extract actual script content."""
    if not text:
        return text
    # 1. Remove <think>...</think> blocks
    text = _THINK_RE.sub("", text)
    # 2. If delimiter exists, take only the content after it
    if GENERATION_DELIMITER in text:
        text = text.split(GENERATION_DELIMITER, 1)[1]
    # 3. Remove any 【...】 bracket headers (prompt section echo)
    text = _BRACKET_HEADER_RE.sub("", text)
    # 4. Strip leading metadata lines that are prompt echo
    lines = text.split("\n")
    cleaned = []
    skipping = True
    for line in lines:
        s = line.strip()
        if skipping and (not s or s.startswith(_PROMPT_META_PREFIXES)):
            continue
        skipping = False
        cleaned.append(line)
    # 5. Truncate trailing noise after "---" separator
    #    Small models echo prompt metadata or add structured analysis after "---".
    #    Once we have enough real content, cut at the first trailing "---".
    result = []
    content_chars = 0
    for line in cleaned:
        s = line.strip()
        if s == "---" and content_chars >= 40:
            break
        result.append(line)
        content_chars += len(s)
    return "\n".join(result).strip()

_RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_FALLBACKABLE_HTTP_STATUS = _RETRYABLE_HTTP_STATUS.union({400, 404, 422})
_NO_FALLBACK_HTTP_STATUS = {401, 403}


class LLMCallError(RuntimeError):
    """统一LLM调用错误模型，用于重试和fallback判定"""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        fallback_eligible: bool,
        status_code: Optional[int] = None,
        backend_name: str = "",
        category: str = "",
    ):
        super().__init__(message)
        self.retryable = retryable
        self.fallback_eligible = fallback_eligible
        self.status_code = status_code
        self.backend_name = backend_name
        self.category = category


class CircuitOpenError(LLMCallError):
    """断路器开启时抛出"""

    def __init__(self, backend_name: str):
        super().__init__(
            f"circuit open for backend={backend_name}",
            retryable=False,
            fallback_eligible=True,
            backend_name=backend_name,
        )


@dataclass
class _LayerPlan:
    name: str
    backend: "LLMBackend"
    category: str
    breaker: "CircuitBreaker"


class CircuitBreaker:
    """轻量异步断路器"""

    def __init__(
        self,
        name: str,
        *,
        enabled: bool,
        failure_threshold: int,
        recovery_seconds: float,
        half_open_max_calls: int,
    ):
        self.name = name
        self.enabled = enabled
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_seconds = max(1.0, recovery_seconds)
        self.half_open_max_calls = max(1, half_open_max_calls)
        self._lock = asyncio.Lock()
        self._state = "closed"  # closed/open/half_open
        self._failure_count = 0
        self._opened_at = 0.0
        self._half_open_inflight = 0

    async def before_call(self):
        if not self.enabled:
            return
        async with self._lock:
            now = time.monotonic()
            if self._state == "open":
                if now - self._opened_at >= self.recovery_seconds:
                    self._state = "half_open"
                    self._half_open_inflight = 0
                else:
                    raise CircuitOpenError(self.name)

            if self._state == "half_open":
                if self._half_open_inflight >= self.half_open_max_calls:
                    raise CircuitOpenError(self.name)
                self._half_open_inflight += 1

    async def record_success(self):
        if not self.enabled:
            return
        async with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._opened_at = 0.0
            self._half_open_inflight = 0

    async def record_failure(self):
        if not self.enabled:
            return
        async with self._lock:
            now = time.monotonic()
            if self._state == "half_open":
                self._state = "open"
                self._opened_at = now
                self._half_open_inflight = 0
                self._failure_count = self.failure_threshold
                return

            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                self._opened_at = now

    async def release_half_open_slot(self):
        if not self.enabled:
            return
        async with self._lock:
            if self._state == "half_open" and self._half_open_inflight > 0:
                self._half_open_inflight -= 1


class LLMBackend(ABC):
    """LLM后端抽象基类 — 内置连接池管理"""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取复用的 aiohttp Session (懒初始化)"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None),
                connector=connector,
            )
        return self._session

    async def _ensure_success(self, resp: aiohttp.ClientResponse, backend_name: str):
        if resp.status < 400:
            return

        detail = ""
        try:
            detail = (await resp.text())[:300]
        except Exception:
            detail = "<unreadable>"

        retryable = resp.status in _RETRYABLE_HTTP_STATUS
        fallback_eligible = (
            resp.status in _FALLBACKABLE_HTTP_STATUS
            and resp.status not in _NO_FALLBACK_HTTP_STATUS
        )
        try:
            resp.raise_for_status()
        except aiohttp.ClientResponseError as exc:
            raise LLMCallError(
                (
                    f"{backend_name} request failed status={resp.status}, "
                    f"message={detail}"
                ),
                retryable=retryable,
                fallback_eligible=fallback_eligible,
                status_code=resp.status,
                backend_name=backend_name,
            ) from exc
        raise LLMCallError(
            f"{backend_name} request failed status={resp.status}, message={detail}",
            retryable=retryable,
            fallback_eligible=fallback_eligible,
            status_code=resp.status,
            backend_name=backend_name,
        )

    async def close(self):
        """关闭连接池"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @abstractmethod
    async def generate(self, prompt: str, category: str = "通用",
                       stream: bool = True,
                       **kwargs) -> AsyncGenerator[str, None]:
        yield ""  # pragma: no cover

    @abstractmethod
    async def generate_sync(self, prompt: str, category: str = "通用",
                            max_tokens: int = 1024, **kwargs) -> str:
        ...

    @abstractmethod
    async def health_check(self) -> Dict:
        ...


class VLLMBackend(LLMBackend):
    """
    vLLM后端 - 通过model参数动态切换LoRA Adapter
    生产环境使用, 支持: PagedAttention, 多LoRA, Prefix Caching
    """

    def __init__(self, base_url: str, model: str,
                 adapter_map: Dict[str, str]):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.base_model = model
        self.adapter_map = adapter_map

    def _get_model_name(self, category: str) -> str:
        return self.adapter_map.get(category, self.base_model)

    async def generate(self, prompt: str, category: str = "通用",
                       stream: bool = True, **kwargs) -> AsyncGenerator[str, None]:
        model = self._get_model_name(category)
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", settings.llm.max_tokens),
            "temperature": kwargs.get("temperature", settings.llm.temperature),
            "top_p": kwargs.get("top_p", settings.llm.top_p),
            "stream": stream,
        }
        timeout = kwargs.get("timeout")
        headers = kwargs.get("headers")

        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/completions",
            json=payload,
            timeout=timeout,
            headers=headers,
        ) as resp:
            await self._ensure_success(resp, "vllm")
            if stream:
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError as exc:
                            raise LLMCallError(
                                f"vllm stream payload decode failed: {exc}",
                                retryable=True,
                                fallback_eligible=True,
                                backend_name="vllm",
                                category=category,
                            ) from exc
                        text = data.get("choices", [{}])[0].get("text", "")
                        if text:
                            yield text
            else:
                data = await resp.json(content_type=None)
                yield data["choices"][0]["text"]

    async def generate_sync(self, prompt: str, category: str = "通用",
                            max_tokens: int = 1024, **kwargs) -> str:
        result = ""
        async for token in self.generate(
            prompt, category, stream=False, max_tokens=max_tokens, **kwargs
        ):
            result += token
        return result

    async def health_check(self) -> Dict:
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/models") as resp:
                await self._ensure_success(resp, "vllm")
                return {"status": "healthy", "models": await resp.json()}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class OllamaBackend(LLMBackend):
    """
    Ollama后端 - 每个垂类是独立模型 (LoRA已合并)
    开发/测试环境使用, 一行命令启动
    """

    def __init__(self, base_url: str, model_map: Dict[str, str]):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model_map = model_map

    def _get_model_name(self, category: str) -> str:
        return self.model_map.get(category, "qwen2.5:0.5b")

    async def generate(self, prompt: str, category: str = "通用",
                       stream: bool = True, **kwargs) -> AsyncGenerator[str, None]:
        model_name = self._get_model_name(category)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", settings.llm.temperature),
                "top_p": kwargs.get("top_p", settings.llm.top_p),
                "num_predict": kwargs.get("max_tokens", settings.llm.max_tokens),
            },
        }
        timeout = kwargs.get("timeout")
        headers = kwargs.get("headers")

        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=timeout,
            headers=headers,
        ) as resp:
            await self._ensure_success(resp, "ollama")
            if stream:
                async for line in resp.content:
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError as exc:
                        raise LLMCallError(
                            f"ollama stream payload decode failed: {exc}",
                            retryable=True,
                            fallback_eligible=True,
                            backend_name="ollama",
                            category=category,
                        ) from exc
                    if not data.get("done"):
                        yield data.get("response", "")
            else:
                data = await resp.json(content_type=None)
                yield data.get("response", "")

    async def generate_sync(self, prompt: str, category: str = "通用",
                            max_tokens: int = 1024, **kwargs) -> str:
        result = ""
        async for token in self.generate(
            prompt, category, stream=False, max_tokens=max_tokens, **kwargs
        ):
            result += token
        return result

    async def health_check(self) -> Dict:
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as resp:
                await self._ensure_success(resp, "ollama")
                return {"status": "healthy", "models": await resp.json()}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class ZhipuBackend(LLMBackend):
    """
    智谱大模型后端（OpenAI-compatible Chat Completions）
    文档接口: /chat/completions
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers

    def _extract_text(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    async def generate(
        self,
        prompt: str,
        category: str = "通用",
        stream: bool = True,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            raise LLMCallError(
                "zhipu api key missing, set ZHIPU_API_KEY",
                retryable=False,
                fallback_eligible=False,
                backend_name="zhipu",
                category=category,
            )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", settings.llm.temperature),
            "top_p": kwargs.get("top_p", settings.llm.top_p),
            "max_tokens": kwargs.get("max_tokens", settings.llm.max_tokens),
            "stream": stream,
        }
        timeout = kwargs.get("timeout")
        headers = self._headers(kwargs.get("headers"))

        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=timeout,
            headers=headers,
        ) as resp:
            await self._ensure_success(resp, "zhipu")
            if not stream:
                data = await resp.json(content_type=None)
                yield self._extract_text(data)
                return

            async for line in resp.content:
                raw = line.decode("utf-8").strip()
                if not raw or not raw.startswith("data:"):
                    continue
                body = raw[5:].strip()
                if body == "[DONE]":
                    return
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise LLMCallError(
                        f"zhipu stream payload decode failed: {exc}",
                        retryable=True,
                        fallback_eligible=True,
                        backend_name="zhipu",
                        category=category,
                    ) from exc
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield str(token)

    async def generate_sync(
        self,
        prompt: str,
        category: str = "通用",
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        result = ""
        async for token in self.generate(
            prompt, category, stream=False, max_tokens=max_tokens, **kwargs
        ):
            result += token
        return result

    async def health_check(self) -> Dict:
        if not self.api_key:
            return {"status": "unhealthy", "error": "missing ZHIPU_API_KEY"}
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/models",
                headers=self._headers(),
            ) as resp:
                await self._ensure_success(resp, "zhipu")
                return {"status": "healthy", "models": await resp.json()}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class LLMServiceClient:
    """
    统一LLM调用入口 - 业务代码不关心底层是vLLM还是Ollama
    通过环境变量自动切换后端
    """

    def __init__(self, env: Optional[str] = None):
        env = env or settings.llm.env
        self._fallback_backend: Optional[LLMBackend] = None
        if env == "production":
            self.backend: LLMBackend = VLLMBackend(
                base_url=settings.llm.vllm_base_url,
                model=settings.llm.vllm_model,
                adapter_map=settings.llm.vllm_adapter_map,
            )
        else:
            self.backend = OllamaBackend(
                base_url=settings.llm.ollama_base_url,
                model_map=settings.llm.ollama_model_map,
            )
        self._fallback_backend = self._build_fallback_backend()
        self._retry_max_attempts = max(1, settings.llm.retry_max_attempts)
        self._retry_base_delay_seconds = max(0.0, settings.llm.retry_base_delay_seconds)
        self._retry_max_delay_seconds = max(
            self._retry_base_delay_seconds, settings.llm.retry_max_delay_seconds
        )
        self._retry_jitter_seconds = max(0.0, settings.llm.retry_jitter_seconds)
        self._idempotency_salt = settings.llm.idempotency_salt
        self._sync_inflight_enabled = settings.llm.idempotency_inflight_enabled
        self._sync_inflight: Dict[str, asyncio.Task[str]] = {}
        self._sync_inflight_lock = asyncio.Lock()
        self._primary_breaker = CircuitBreaker(
            f"primary:{type(self.backend).__name__}",
            enabled=settings.llm.circuit_breaker_enabled,
            failure_threshold=settings.llm.circuit_breaker_failure_threshold,
            recovery_seconds=settings.llm.circuit_breaker_recovery_seconds,
            half_open_max_calls=settings.llm.circuit_breaker_half_open_max_calls,
        )
        self._fallback_breaker = CircuitBreaker(
            f"fallback:{type(self._fallback_backend).__name__}"
            if self._fallback_backend is not None else "fallback:none",
            enabled=settings.llm.circuit_breaker_enabled,
            failure_threshold=settings.llm.circuit_breaker_failure_threshold,
            recovery_seconds=settings.llm.circuit_breaker_recovery_seconds,
            half_open_max_calls=settings.llm.circuit_breaker_half_open_max_calls,
        )
        logger.info(f"LLMServiceClient initialized with backend: {type(self.backend).__name__}")

    def _build_fallback_backend(self) -> Optional[LLMBackend]:
        if not settings.llm.fallback_enabled:
            return None

        backend_type = settings.llm.fallback_backend.lower()
        if backend_type == "vllm":
            return VLLMBackend(
                base_url=settings.llm.fallback_base_url,
                model=settings.llm.fallback_model,
                adapter_map={"通用": settings.llm.fallback_model},
            )
        if backend_type == "ollama":
            return OllamaBackend(
                base_url=settings.llm.fallback_base_url,
                model_map={"通用": settings.llm.fallback_model},
            )
        if backend_type == "zhipu":
            return ZhipuBackend(
                base_url=settings.llm.zhipu_base_url,
                api_key=settings.llm.zhipu_api_key,
                model=settings.llm.zhipu_model,
            )

        logger.warning("Unsupported fallback backend: %s", backend_type)
        return None

    def _build_timeout(self, stream: bool, layer_index: int) -> aiohttp.ClientTimeout:
        connect = settings.llm.timeout_connect_seconds
        if stream:
            total = settings.llm.timeout_total_stream_seconds
            read = settings.llm.timeout_read_stream_seconds
        else:
            total = settings.llm.timeout_total_sync_seconds
            read = settings.llm.timeout_read_sync_seconds

        # fallback层使用更短超时，防止雪崩等待
        if layer_index > 0:
            total = max(connect + 1.0, total * settings.llm.fallback_timeout_factor)
            read = max(1.0, read * settings.llm.fallback_timeout_factor)

        return aiohttp.ClientTimeout(total=total, connect=connect, sock_read=read)

    def _build_request_headers(
        self,
        idempotency_key: str,
        layer_name: str,
        attempt: int,
        stream: bool,
        request_id: Optional[str] = None,
    ) -> Dict[str, str]:
        headers = {
            "Idempotency-Key": idempotency_key,
            "X-LLM-Layer": layer_name,
            "X-LLM-Attempt": str(attempt),
            "X-LLM-Mode": "stream" if stream else "sync",
        }
        if request_id:
            headers["X-Request-Id"] = request_id
        return headers

    def _resolve_idempotency_key(
        self,
        prompt: str,
        category: str,
        request_id: Optional[str],
        explicit_key: Optional[str],
    ) -> str:
        if explicit_key:
            return explicit_key
        digest = hashlib.sha256(
            f"{self._idempotency_salt}|{category}|{request_id or ''}|{prompt}".encode("utf-8")
        ).hexdigest()
        return digest[:40]

    def _retry_delay(self, attempt: int) -> float:
        delay = min(
            self._retry_max_delay_seconds,
            self._retry_base_delay_seconds * (2 ** max(0, attempt - 1)),
        )
        if self._retry_jitter_seconds > 0:
            delay += random.uniform(0.0, self._retry_jitter_seconds)
        return delay

    def _classify_exception(
        self, exc: Exception, backend_name: str, category: str
    ) -> LLMCallError:
        if isinstance(exc, LLMCallError):
            if not exc.backend_name:
                exc.backend_name = backend_name
            if not exc.category:
                exc.category = category
            return exc
        if isinstance(exc, asyncio.TimeoutError):
            return LLMCallError(
                f"timeout from backend={backend_name}",
                retryable=True,
                fallback_eligible=True,
                backend_name=backend_name,
                category=category,
            )
        if isinstance(exc, aiohttp.ClientConnectionError):
            return LLMCallError(
                f"connection error from backend={backend_name}: {exc}",
                retryable=True,
                fallback_eligible=True,
                backend_name=backend_name,
                category=category,
            )
        if isinstance(exc, aiohttp.ClientPayloadError):
            return LLMCallError(
                f"payload error from backend={backend_name}: {exc}",
                retryable=True,
                fallback_eligible=True,
                backend_name=backend_name,
                category=category,
            )
        if isinstance(exc, aiohttp.ClientResponseError):
            status = exc.status
            retryable = status in _RETRYABLE_HTTP_STATUS
            fallback_eligible = (
                status in _FALLBACKABLE_HTTP_STATUS and status not in _NO_FALLBACK_HTTP_STATUS
            )
            return LLMCallError(
                f"http error from backend={backend_name}, status={status}",
                retryable=retryable,
                fallback_eligible=fallback_eligible,
                status_code=status,
                backend_name=backend_name,
                category=category,
            )
        return LLMCallError(
            f"unexpected llm error from backend={backend_name}: {exc}",
            retryable=False,
            fallback_eligible=False,
            backend_name=backend_name,
            category=category,
        )

    def _should_retry(self, err: LLMCallError, attempt: int) -> bool:
        return err.retryable and attempt < self._retry_max_attempts

    def _can_try_next_layer(self, err: LLMCallError) -> bool:
        return err.fallback_eligible

    def _build_layer_plans(
        self,
        category: str,
        prefer_fallback: bool = False,
    ) -> Tuple[_LayerPlan, ...]:
        fallback_plan: Optional[_LayerPlan] = None
        if self._fallback_backend is not None:
            fallback_plan = _LayerPlan(
                name="fallback-backend",
                backend=self._fallback_backend,
                category=category if settings.llm.fallback_keep_category else "通用",
                breaker=self._fallback_breaker,
            )

        if prefer_fallback and fallback_plan is not None:
            return (fallback_plan,)

        layers = [
            _LayerPlan(
                name="primary",
                backend=self.backend,
                category=category,
                breaker=self._primary_breaker,
            )
        ]
        if settings.llm.fallback_to_general_enabled and category != "通用":
            layers.append(
                _LayerPlan(
                    name="primary-general",
                    backend=self.backend,
                    category="通用",
                    breaker=self._primary_breaker,
                )
            )
        if fallback_plan is not None:
            layers.append(fallback_plan)
        return tuple(layers)

    async def _run_sync_idempotent(
        self, idempotency_key: str, coro_factory
    ) -> str:
        if not self._sync_inflight_enabled:
            return await coro_factory()

        owner = False
        async with self._sync_inflight_lock:
            fut = self._sync_inflight.get(idempotency_key)
            if fut is None:
                fut = asyncio.create_task(coro_factory())
                self._sync_inflight[idempotency_key] = fut
                owner = True

        try:
            return await fut
        finally:
            if owner:
                async with self._sync_inflight_lock:
                    self._sync_inflight.pop(idempotency_key, None)

    async def _call_sync_with_layer(
        self,
        plan: _LayerPlan,
        layer_index: int,
        prompt: str,
        max_tokens: int,
        idempotency_key: str,
        request_id: Optional[str],
        kwargs: Dict,
    ) -> str:
        last_error: Optional[LLMCallError] = None
        for attempt in range(1, self._retry_max_attempts + 1):
            entered_breaker = False
            try:
                await plan.breaker.before_call()
                entered_breaker = True
                timeout = self._build_timeout(stream=False, layer_index=layer_index)
                headers = self._build_request_headers(
                    idempotency_key=idempotency_key,
                    layer_name=plan.name,
                    attempt=attempt,
                    stream=False,
                    request_id=request_id,
                )
                result = await plan.backend.generate_sync(
                    prompt,
                    plan.category,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    headers=headers,
                    idempotency_key=idempotency_key,
                    request_id=request_id,
                    **kwargs,
                )
                await plan.breaker.record_success()
                return result
            except Exception as exc:
                err = self._classify_exception(exc, plan.name, plan.category)
                last_error = err
                if entered_breaker:
                    await plan.breaker.record_failure()
                if self._should_retry(err, attempt):
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise err
            finally:
                if entered_breaker:
                    await plan.breaker.release_half_open_slot()
        if last_error is None:
            last_error = LLMCallError(
                f"sync generation failed in unknown state for layer={plan.name}",
                retryable=False,
                fallback_eligible=False,
                backend_name=plan.name,
                category=plan.category,
            )
        raise last_error

    async def _call_stream_with_layer(
        self,
        plan: _LayerPlan,
        layer_index: int,
        prompt: str,
        idempotency_key: str,
        request_id: Optional[str],
        kwargs: Dict,
    ) -> AsyncGenerator[str, None]:
        for attempt in range(1, self._retry_max_attempts + 1):
            entered_breaker = False
            emitted = False
            try:
                await plan.breaker.before_call()
                entered_breaker = True
                timeout = self._build_timeout(stream=True, layer_index=layer_index)
                headers = self._build_request_headers(
                    idempotency_key=idempotency_key,
                    layer_name=plan.name,
                    attempt=attempt,
                    stream=True,
                    request_id=request_id,
                )
                async for token in plan.backend.generate(
                    prompt,
                    plan.category,
                    True,
                    timeout=timeout,
                    headers=headers,
                    idempotency_key=idempotency_key,
                    request_id=request_id,
                    **kwargs,
                ):
                    emitted = True
                    yield token
                await plan.breaker.record_success()
                return
            except Exception as exc:
                err = self._classify_exception(exc, plan.name, plan.category)
                if entered_breaker:
                    await plan.breaker.record_failure()
                if emitted:
                    raise err
                if self._should_retry(err, attempt):
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise err
            finally:
                if entered_breaker:
                    await plan.breaker.release_half_open_slot()

    async def generate(self, prompt: str, category: str = "通用",
                       stream: bool = True,
                       **kwargs) -> AsyncGenerator[str, None]:
        """统一流式生成接口"""
        if not stream:
            text = await self.generate_sync(prompt, category, **kwargs)
            if text:
                yield text
            return

        request_id = kwargs.pop("request_id", None)
        explicit_key = kwargs.pop("idempotency_key", None)
        prefer_fallback = bool(kwargs.pop("prefer_fallback", False))
        idempotency_key = self._resolve_idempotency_key(
            prompt=prompt,
            category=category,
            request_id=request_id,
            explicit_key=explicit_key,
        )
        plans = self._build_layer_plans(category, prefer_fallback=prefer_fallback)
        last_error: Optional[LLMCallError] = None

        for idx, plan in enumerate(plans):
            layer_emitted = False
            try:
                async for token in self._call_stream_with_layer(
                    plan=plan,
                    layer_index=idx,
                    prompt=prompt,
                    idempotency_key=idempotency_key,
                    request_id=request_id,
                    kwargs=kwargs,
                ):
                    layer_emitted = True
                    yield token
                return
            except LLMCallError as err:
                last_error = err
                if layer_emitted or not self._can_try_next_layer(err) or idx == len(plans) - 1:
                    raise
                logger.warning(
                    "LLM stream layer failed, switch next layer. layer=%s category=%s err=%s",
                    plan.name,
                    plan.category,
                    err,
                )

        if last_error is not None:
            raise last_error

    async def generate_sync(self, prompt: str, category: str = "通用",
                            max_tokens: int = 1024, **kwargs) -> str:
        """统一同步生成接口"""
        request_id = kwargs.pop("request_id", None)
        explicit_key = kwargs.pop("idempotency_key", None)
        prefer_fallback = bool(kwargs.pop("prefer_fallback", False))
        idempotency_key = self._resolve_idempotency_key(
            prompt=prompt,
            category=category,
            request_id=request_id,
            explicit_key=explicit_key,
        )
        plans = self._build_layer_plans(category, prefer_fallback=prefer_fallback)

        async def _run_layers() -> str:
            last_error: Optional[LLMCallError] = None
            for idx, plan in enumerate(plans):
                try:
                    return await self._call_sync_with_layer(
                        plan=plan,
                        layer_index=idx,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        idempotency_key=idempotency_key,
                        request_id=request_id,
                        kwargs=kwargs,
                    )
                except LLMCallError as err:
                    last_error = err
                    if not self._can_try_next_layer(err) or idx == len(plans) - 1:
                        raise
                    logger.warning(
                        "LLM sync layer failed, switch next layer. layer=%s category=%s err=%s",
                        plan.name,
                        plan.category,
                        err,
                    )
            if last_error is None:
                raise LLMCallError(
                    "llm sync generation failed with unknown reason",
                    retryable=False,
                    fallback_eligible=False,
                )
            raise last_error

        return await self._run_sync_idempotent(idempotency_key, _run_layers)

    async def health_check(self) -> Dict:
        return await self.backend.health_check()

    async def close(self):
        """关闭后端连接"""
        await self.backend.close()
        if self._fallback_backend is not None:
            await self._fallback_backend.close()
