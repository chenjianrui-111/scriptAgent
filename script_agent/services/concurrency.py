"""
并发控制组件

支持两种会话锁:
  - SessionLockManager: 进程内锁
  - RedisSessionLockManager: Redis 分布式锁 (多实例一致)
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

from script_agent.config.settings import settings

logger = logging.getLogger(__name__)


class SessionLockTimeoutError(TimeoutError):
    """会话锁获取超时"""


class BaseSessionLockManager:
    """会话锁管理器抽象"""

    @asynccontextmanager
    async def acquire(
        self,
        session_id: str,
        timeout_seconds: Optional[float] = None,
    ) -> AsyncIterator[None]:
        raise NotImplementedError

    async def stats(self) -> Dict[str, Any]:
        return {}

    async def close(self) -> None:
        return None


@dataclass
class _LockEntry:
    lock: asyncio.Lock
    waiters: int = 0
    last_used_at: float = 0.0


class SessionLockManager(BaseSessionLockManager):
    """会话级进程内异步锁管理器"""

    def __init__(self, default_timeout_seconds: float = 8.0):
        self._default_timeout_seconds = default_timeout_seconds
        self._entries: Dict[str, _LockEntry] = {}
        self._manager_lock = asyncio.Lock()

    async def _get_or_create_entry(self, session_id: str) -> _LockEntry:
        async with self._manager_lock:
            entry = self._entries.get(session_id)
            if entry is None:
                entry = _LockEntry(lock=asyncio.Lock(), last_used_at=time.time())
                self._entries[session_id] = entry
            entry.last_used_at = time.time()
            return entry

    async def _cleanup_entry_if_idle(self, session_id: str) -> None:
        async with self._manager_lock:
            entry = self._entries.get(session_id)
            if entry and (not entry.lock.locked()) and entry.waiters == 0:
                self._entries.pop(session_id, None)

    async def stats(self) -> Dict[str, Any]:
        async with self._manager_lock:
            total = len(self._entries)
            active = 0
            queued = 0
            for entry in self._entries.values():
                if entry.lock.locked():
                    active += 1
                queued += entry.waiters
            return {
                "backend": "local",
                "tracked_sessions": total,
                "active_locks": active,
                "queued_waiters": queued,
                "default_timeout_seconds": self._default_timeout_seconds,
            }

    @asynccontextmanager
    async def acquire(
        self,
        session_id: str,
        timeout_seconds: Optional[float] = None,
    ) -> AsyncIterator[None]:
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._default_timeout_seconds
        )
        entry = await self._get_or_create_entry(session_id)
        entry.waiters += 1
        try:
            await asyncio.wait_for(entry.lock.acquire(), timeout=timeout)
        except TimeoutError as exc:
            raise SessionLockTimeoutError(
                f"failed to acquire lock for session '{session_id}' "
                f"within {timeout:.1f}s"
            ) from exc
        finally:
            entry.waiters = max(entry.waiters - 1, 0)

        try:
            yield
        finally:
            if entry.lock.locked():
                entry.lock.release()
            entry.last_used_at = time.time()
            await self._cleanup_entry_if_idle(session_id)


class RedisSessionLockManager(BaseSessionLockManager):
    """Redis 分布式会话锁 (SET NX PX)"""

    _RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
else
  return 0
end
"""

    def __init__(
        self,
        redis_url: str,
        default_timeout_seconds: float = 8.0,
        key_prefix: str = "script_agent:lock:session:",
        lease_seconds: int = 30,
        retry_interval_ms: int = 120,
    ):
        self._redis_url = redis_url
        self._default_timeout_seconds = default_timeout_seconds
        self._key_prefix = key_prefix
        self._lease_seconds = lease_seconds
        self._retry_interval_ms = retry_interval_ms
        self._redis = None

        self._meta_lock = asyncio.Lock()
        self._active_locks = 0
        self._waiters = 0
        self._degraded_to_local = False
        self._local_fallback = SessionLockManager(
            default_timeout_seconds=default_timeout_seconds
        )

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as exc:
                raise RuntimeError(
                    "redis package required for distributed locks"
                ) from exc
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    async def stats(self) -> Dict[str, Any]:
        local_stats = await self._local_fallback.stats()
        async with self._meta_lock:
            return {
                "backend": "redis",
                "active_locks": self._active_locks,
                "queued_waiters": self._waiters,
                "default_timeout_seconds": self._default_timeout_seconds,
                "lease_seconds": self._lease_seconds,
                "retry_interval_ms": self._retry_interval_ms,
                "degraded_to_local": self._degraded_to_local,
                "local_fallback": local_stats,
            }

    @asynccontextmanager
    async def acquire(
        self,
        session_id: str,
        timeout_seconds: Optional[float] = None,
    ) -> AsyncIterator[None]:
        if self._degraded_to_local:
            async with self._local_fallback.acquire(session_id, timeout_seconds):
                yield
            return

        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._default_timeout_seconds
        )
        deadline = time.monotonic() + timeout
        key = self._key(session_id)
        token = str(uuid.uuid4())
        acquired = False

        async with self._meta_lock:
            self._waiters += 1

        try:
            try:
                redis_client = await self._get_redis()
                while True:
                    result = await redis_client.set(
                        key,
                        token,
                        nx=True,
                        px=int(self._lease_seconds * 1000),
                    )
                    if result:
                        acquired = True
                        async with self._meta_lock:
                            self._active_locks += 1
                        break

                    if time.monotonic() >= deadline:
                        raise SessionLockTimeoutError(
                            f"failed to acquire distributed lock for session "
                            f"'{session_id}' within {timeout:.1f}s"
                        )

                    await asyncio.sleep(self._retry_interval_ms / 1000)
            except Exception as exc:
                logger.warning(
                    "Redis distributed lock unavailable, degrade to local lock: %s",
                    exc,
                )
                self._degraded_to_local = True
                async with self._local_fallback.acquire(session_id, timeout_seconds):
                    yield
                return
        finally:
            async with self._meta_lock:
                self._waiters = max(self._waiters - 1, 0)

        try:
            yield
        finally:
            if acquired:
                try:
                    redis_client = await self._get_redis()
                    await redis_client.eval(self._RELEASE_SCRIPT, 1, key, token)
                except Exception as exc:
                    logger.warning("Failed to release distributed lock: %s", exc)
                async with self._meta_lock:
                    self._active_locks = max(self._active_locks - 1, 0)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        await self._local_fallback.close()


def create_session_lock_manager() -> BaseSessionLockManager:
    """根据配置创建会话锁管理器"""
    if settings.orchestration.distributed_lock_enabled:
        try:
            import redis.asyncio  # type: ignore # noqa: F401
            return RedisSessionLockManager(
                redis_url=settings.cache.redis_url,
                default_timeout_seconds=settings.orchestration.session_lock_timeout_seconds,
                key_prefix=settings.orchestration.distributed_lock_prefix,
                lease_seconds=settings.orchestration.distributed_lock_lease_seconds,
                retry_interval_ms=settings.orchestration.distributed_lock_retry_interval_ms,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize RedisSessionLockManager, fallback local lock: %s",
                exc,
            )

    return SessionLockManager(
        default_timeout_seconds=settings.orchestration.session_lock_timeout_seconds
    )
