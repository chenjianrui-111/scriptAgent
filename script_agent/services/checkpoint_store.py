"""
工作流 Checkpoint 独立存储

能力:
  - 版本化写入 (version 自增)
  - 最新快照读取
  - 历史回放 (按 version 查询)
  - 审计字段 (trace_id/status/checksum/created_at)
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from script_agent.config.settings import settings

logger = logging.getLogger(__name__)


class CheckpointStore(ABC):
    @abstractmethod
    async def append(
        self,
        session_id: str,
        payload: Dict[str, Any],
        trace_id: str,
        status: str,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def latest(self, session_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    async def get_version(self, session_id: str, version: int) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    async def history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        ...

    async def close(self) -> None:
        return None

    async def stats(self) -> Dict[str, Any]:
        return {}


class MemoryCheckpointStore(CheckpointStore):
    def __init__(self):
        self._data: Dict[str, List[Dict[str, Any]]] = {}

    async def append(
        self,
        session_id: str,
        payload: Dict[str, Any],
        trace_id: str,
        status: str,
    ) -> Dict[str, Any]:
        records = self._data.setdefault(session_id, [])
        version = len(records) + 1
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        checksum = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        record = {
            "session_id": session_id,
            "version": version,
            "trace_id": trace_id,
            "status": status,
            "created_at": datetime.now().isoformat(),
            "checksum": checksum,
            "payload": json.loads(payload_json),
        }
        records.append(record)
        return record

    async def latest(self, session_id: str) -> Optional[Dict[str, Any]]:
        records = self._data.get(session_id, [])
        return records[-1] if records else None

    async def get_version(self, session_id: str, version: int) -> Optional[Dict[str, Any]]:
        records = self._data.get(session_id, [])
        for record in records:
            if record["version"] == version:
                return record
        return None

    async def history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        records = self._data.get(session_id, [])
        return list(reversed(records[-limit:]))

    async def stats(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "sessions": len(self._data),
            "records": sum(len(v) for v in self._data.values()),
        }


class RedisCheckpointStore(CheckpointStore):
    def __init__(self, redis_url: str, prefix: str):
        self._redis_url = redis_url
        self._prefix = prefix
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as exc:
                raise RuntimeError(
                    "redis package required for redis checkpoint store"
                ) from exc
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def _seq_key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}:seq"

    def _idx_key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}:idx"

    def _entry_key(self, session_id: str, version: int) -> str:
        return f"{self._prefix}{session_id}:v:{version}"

    async def append(
        self,
        session_id: str,
        payload: Dict[str, Any],
        trace_id: str,
        status: str,
    ) -> Dict[str, Any]:
        redis_client = await self._get_redis()
        version = int(await redis_client.incr(self._seq_key(session_id)))
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        checksum = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

        record = {
            "session_id": session_id,
            "version": version,
            "trace_id": trace_id,
            "status": status,
            "created_at": datetime.now().isoformat(),
            "checksum": checksum,
            "payload": json.loads(payload_json),
        }
        record_json = json.dumps(record, ensure_ascii=False)

        pipe = redis_client.pipeline(transaction=True)
        pipe.set(self._entry_key(session_id, version), record_json)
        pipe.zadd(self._idx_key(session_id), {str(version): version})
        await pipe.execute()
        return record

    async def latest(self, session_id: str) -> Optional[Dict[str, Any]]:
        redis_client = await self._get_redis()
        versions = await redis_client.zrevrange(self._idx_key(session_id), 0, 0)
        if not versions:
            return None
        raw = await redis_client.get(self._entry_key(session_id, int(versions[0])))
        return json.loads(raw) if raw else None

    async def get_version(self, session_id: str, version: int) -> Optional[Dict[str, Any]]:
        redis_client = await self._get_redis()
        raw = await redis_client.get(self._entry_key(session_id, version))
        return json.loads(raw) if raw else None

    async def history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        redis_client = await self._get_redis()
        versions = await redis_client.zrevrange(self._idx_key(session_id), 0, max(limit - 1, 0))
        if not versions:
            return []
        keys = [self._entry_key(session_id, int(v)) for v in versions]
        raws = await redis_client.mget(keys)
        return [json.loads(r) for r in raws if r]

    async def stats(self) -> Dict[str, Any]:
        return {
            "backend": "redis",
            "prefix": self._prefix,
        }

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None


class SQLiteCheckpointStore(CheckpointStore):
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db = None

    async def _get_db(self):
        if self._db is None:
            try:
                import aiosqlite
            except ImportError as exc:
                raise RuntimeError(
                    "aiosqlite package required for sqlite checkpoint store"
                ) from exc

            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                    session_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    trace_id TEXT,
                    status TEXT,
                    payload_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, version)
                )
                """
            )
            await self._db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_session
                ON workflow_checkpoints(session_id, version DESC)
                """
            )
            await self._db.commit()
        return self._db

    async def append(
        self,
        session_id: str,
        payload: Dict[str, Any],
        trace_id: str,
        status: str,
    ) -> Dict[str, Any]:
        db = await self._get_db()
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        checksum = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        created_at = datetime.now().isoformat()

        async with db.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM workflow_checkpoints WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            version = int(row[0] if row else 1)

        await db.execute(
            """
            INSERT INTO workflow_checkpoints (
                session_id, version, trace_id, status, payload_json, checksum, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, version, trace_id, status, payload_json, checksum, created_at),
        )
        await db.commit()

        return {
            "session_id": session_id,
            "version": version,
            "trace_id": trace_id,
            "status": status,
            "created_at": created_at,
            "checksum": checksum,
            "payload": json.loads(payload_json),
        }

    async def latest(self, session_id: str) -> Optional[Dict[str, Any]]:
        db = await self._get_db()
        async with db.execute(
            """
            SELECT version, trace_id, status, payload_json, checksum, created_at
            FROM workflow_checkpoints
            WHERE session_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            version, trace_id, status, payload_json, checksum, created_at = row
            return {
                "session_id": session_id,
                "version": int(version),
                "trace_id": trace_id,
                "status": status,
                "created_at": created_at,
                "checksum": checksum,
                "payload": json.loads(payload_json),
            }

    async def get_version(self, session_id: str, version: int) -> Optional[Dict[str, Any]]:
        db = await self._get_db()
        async with db.execute(
            """
            SELECT trace_id, status, payload_json, checksum, created_at
            FROM workflow_checkpoints
            WHERE session_id = ? AND version = ?
            """,
            (session_id, version),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            trace_id, status, payload_json, checksum, created_at = row
            return {
                "session_id": session_id,
                "version": version,
                "trace_id": trace_id,
                "status": status,
                "created_at": created_at,
                "checksum": checksum,
                "payload": json.loads(payload_json),
            }

    async def history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        db = await self._get_db()
        async with db.execute(
            """
            SELECT version, trace_id, status, payload_json, checksum, created_at
            FROM workflow_checkpoints
            WHERE session_id = ?
            ORDER BY version DESC
            LIMIT ?
            """,
            (session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        records = []
        for row in rows:
            version, trace_id, status, payload_json, checksum, created_at = row
            records.append(
                {
                    "session_id": session_id,
                    "version": int(version),
                    "trace_id": trace_id,
                    "status": status,
                    "created_at": created_at,
                    "checksum": checksum,
                    "payload": json.loads(payload_json),
                }
            )
        return records

    async def stats(self) -> Dict[str, Any]:
        return {
            "backend": "sqlite",
            "path": self._db_path,
        }

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


class WorkflowCheckpointManager:
    """checkpoint 管理器: 版本写入 + 历史回放 + 审计查询"""

    def __init__(self, store: Optional[CheckpointStore] = None):
        self._store = store or create_checkpoint_store()

    async def write(
        self,
        session_id: str,
        payload: Dict[str, Any],
        trace_id: str,
        status: str,
    ) -> Dict[str, Any]:
        return await self._store.append(session_id, payload, trace_id, status)

    async def latest_payload(self, session_id: str) -> Optional[Dict[str, Any]]:
        record = await self._store.latest(session_id)
        return record.get("payload") if record else None

    async def latest_record(self, session_id: str) -> Optional[Dict[str, Any]]:
        return await self._store.latest(session_id)

    async def replay(self, session_id: str, version: int) -> Optional[Dict[str, Any]]:
        record = await self._store.get_version(session_id, version)
        return record.get("payload") if record else None

    async def history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(limit, settings.checkpoint.history_limit))
        return await self._store.history(session_id, safe_limit)

    async def close(self) -> None:
        await self._store.close()

    async def stats(self) -> Dict[str, Any]:
        return await self._store.stats()


def create_checkpoint_store() -> CheckpointStore:
    store = settings.checkpoint.store
    if store == "redis":
        try:
            import redis.asyncio  # type: ignore # noqa: F401
            return RedisCheckpointStore(
                redis_url=settings.cache.redis_url,
                prefix=settings.checkpoint.redis_prefix,
            )
        except Exception as exc:
            logger.warning("init RedisCheckpointStore failed, fallback memory: %s", exc)
            return MemoryCheckpointStore()

    if store == "sqlite":
        try:
            import aiosqlite  # type: ignore # noqa: F401
            return SQLiteCheckpointStore(db_path=settings.checkpoint.sqlite_path)
        except Exception as exc:
            logger.warning("init SQLiteCheckpointStore failed, fallback memory: %s", exc)
            return MemoryCheckpointStore()

    return MemoryCheckpointStore()
