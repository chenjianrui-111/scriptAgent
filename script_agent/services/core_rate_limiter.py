"""
核心接口限流: QPS + Token

支持:
  - LocalCoreRateLimiter (单实例)
  - RedisCoreRateLimiter (多实例共享配额)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from script_agent.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RateLimitDecision:
    allowed: bool
    reason: str = ""
    retry_after_seconds: float = 0.0


class BaseCoreRateLimiter:
    async def check_and_consume(self, tenant_id: str, token_cost: int) -> RateLimitDecision:
        raise NotImplementedError

    async def stats(self) -> Dict[str, Any]:
        return {}

    async def close(self) -> None:
        return None


class LocalCoreRateLimiter(BaseCoreRateLimiter):
    def __init__(self, qps_per_tenant: int, tokens_per_minute: int):
        self._qps_per_tenant = qps_per_tenant
        self._tokens_per_minute = tokens_per_minute
        self._lock = asyncio.Lock()
        self._qps_bucket: Dict[str, Dict[int, int]] = {}
        self._token_bucket: Dict[str, Dict[int, int]] = {}

    async def check_and_consume(self, tenant_id: str, token_cost: int) -> RateLimitDecision:
        now = time.time()
        second_slot = int(now)
        minute_slot = int(now // 60)
        token_cost = max(1, token_cost)

        async with self._lock:
            qps_slots = self._qps_bucket.setdefault(tenant_id, {})
            tok_slots = self._token_bucket.setdefault(tenant_id, {})

            # 清理历史窗口
            for s in list(qps_slots.keys()):
                if s < second_slot - 2:
                    qps_slots.pop(s, None)
            for m in list(tok_slots.keys()):
                if m < minute_slot - 2:
                    tok_slots.pop(m, None)

            cur_qps = qps_slots.get(second_slot, 0)
            if cur_qps + 1 > self._qps_per_tenant:
                retry = max(0.0, (second_slot + 1) - now)
                return RateLimitDecision(False, "qps_limit_exceeded", retry)

            cur_tokens = tok_slots.get(minute_slot, 0)
            if cur_tokens + token_cost > self._tokens_per_minute:
                retry = max(0.0, ((minute_slot + 1) * 60) - now)
                return RateLimitDecision(False, "token_limit_exceeded", retry)

            qps_slots[second_slot] = cur_qps + 1
            tok_slots[minute_slot] = cur_tokens + token_cost

        return RateLimitDecision(True)

    async def stats(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "backend": "local",
                "qps_per_tenant": self._qps_per_tenant,
                "tokens_per_minute": self._tokens_per_minute,
                "tracked_tenants": len(self._qps_bucket),
            }


class RedisCoreRateLimiter(BaseCoreRateLimiter):
    def __init__(
        self,
        redis_url: str,
        key_prefix: str,
        qps_per_tenant: int,
        tokens_per_minute: int,
    ):
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._qps_per_tenant = qps_per_tenant
        self._tokens_per_minute = tokens_per_minute
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as exc:
                raise RuntimeError(
                    "redis package required for redis core rate limiter"
                ) from exc
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def _qps_key(self, tenant_id: str, second_slot: int) -> str:
        return f"{self._key_prefix}qps:{tenant_id}:{second_slot}"

    def _tok_key(self, tenant_id: str, minute_slot: int) -> str:
        return f"{self._key_prefix}tok:{tenant_id}:{minute_slot}"

    async def check_and_consume(self, tenant_id: str, token_cost: int) -> RateLimitDecision:
        token_cost = max(1, token_cost)
        now = time.time()
        second_slot = int(now)
        minute_slot = int(now // 60)

        redis_client = await self._get_redis()
        qps_key = self._qps_key(tenant_id, second_slot)
        tok_key = self._tok_key(tenant_id, minute_slot)

        pipe = redis_client.pipeline(transaction=True)
        pipe.incrby(qps_key, 1)
        pipe.expire(qps_key, 2)
        pipe.incrby(tok_key, token_cost)
        pipe.expire(tok_key, 120)
        qps_count, _, token_count, _ = await pipe.execute()

        if int(qps_count) > self._qps_per_tenant:
            retry = max(0.0, (second_slot + 1) - now)
            return RateLimitDecision(False, "qps_limit_exceeded", retry)

        if int(token_count) > self._tokens_per_minute:
            retry = max(0.0, ((minute_slot + 1) * 60) - now)
            return RateLimitDecision(False, "token_limit_exceeded", retry)

        return RateLimitDecision(True)

    async def stats(self) -> Dict[str, Any]:
        return {
            "backend": "redis",
            "qps_per_tenant": self._qps_per_tenant,
            "tokens_per_minute": self._tokens_per_minute,
            "prefix": self._key_prefix,
        }

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None


class CoreRateLimiter:
    def __init__(self, backend: Optional[BaseCoreRateLimiter] = None):
        self._backend = backend or create_core_rate_limiter_backend()

    async def check_and_consume(self, tenant_id: str, token_cost: int) -> RateLimitDecision:
        if not settings.core_rate_limit.enabled:
            return RateLimitDecision(True)
        return await self._backend.check_and_consume(tenant_id, token_cost)

    async def stats(self) -> Dict[str, Any]:
        return await self._backend.stats()

    async def close(self) -> None:
        await self._backend.close()


def create_core_rate_limiter_backend() -> BaseCoreRateLimiter:
    cfg = settings.core_rate_limit
    if cfg.backend == "redis":
        try:
            import redis.asyncio  # type: ignore # noqa: F401
            return RedisCoreRateLimiter(
                redis_url=settings.cache.redis_url,
                key_prefix=cfg.redis_prefix,
                qps_per_tenant=cfg.qps_per_tenant,
                tokens_per_minute=cfg.tokens_per_minute,
            )
        except Exception as exc:
            logger.warning("init RedisCoreRateLimiter failed, fallback local: %s", exc)

    return LocalCoreRateLimiter(
        qps_per_tenant=cfg.qps_per_tenant,
        tokens_per_minute=cfg.tokens_per_minute,
    )
