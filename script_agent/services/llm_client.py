"""
统一LLM调用接口 - 屏蔽vLLM和Ollama的差异

开发环境 → Ollama (本地)
测试环境 → Ollama (服务器)
生产环境 → vLLM (K8s集群, 支持LoRA动态切换)

aiohttp.ClientSession 在后端生命周期内复用, 避免每次请求创建新连接池。
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Optional

import aiohttp

from script_agent.config.settings import settings

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """LLM后端抽象基类 — 内置连接池管理"""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取复用的 aiohttp Session (懒初始化)"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120, connect=10)
            connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout, connector=connector,
            )
        return self._session

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

        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/completions", json=payload,
        ) as resp:
            if stream:
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        text = data.get("choices", [{}])[0].get("text", "")
                        if text:
                            yield text
            else:
                data = await resp.json()
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
        return self.model_map.get(category, "qwen:7b")

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

        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/api/generate", json=payload,
        ) as resp:
            if stream:
                async for line in resp.content:
                    if not line:
                        continue
                    data = json.loads(line)
                    if not data.get("done"):
                        yield data.get("response", "")
            else:
                data = await resp.json()
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

        logger.warning("Unsupported fallback backend: %s", backend_type)
        return None

    async def generate(self, prompt: str, category: str = "通用",
                       stream: bool = True,
                       **kwargs) -> AsyncGenerator[str, None]:
        """统一流式生成接口"""
        emitted = False
        try:
            async for token in self.backend.generate(prompt, category, stream, **kwargs):
                emitted = True
                yield token
            return
        except Exception as exc:
            if emitted or self._fallback_backend is None:
                raise
            logger.warning("Primary LLM stream failed, fallback to backup model: %s", exc)

        async for token in self._fallback_backend.generate(
            prompt, "通用", stream, **kwargs
        ):
            yield token

    async def generate_sync(self, prompt: str, category: str = "通用",
                            max_tokens: int = 1024, **kwargs) -> str:
        """统一同步生成接口"""
        try:
            return await self.backend.generate_sync(
                prompt, category, max_tokens=max_tokens, **kwargs
            )
        except Exception as exc:
            if self._fallback_backend is None:
                raise
            logger.warning(
                "Primary LLM sync failed, fallback to backup model: %s",
                exc,
            )
            return await self._fallback_backend.generate_sync(
                prompt, "通用", max_tokens=max_tokens, **kwargs
            )

    async def health_check(self) -> Dict:
        return await self.backend.health_check()

    async def close(self):
        """关闭后端连接"""
        await self.backend.close()
        if self._fallback_backend is not None:
            await self._fallback_backend.close()
