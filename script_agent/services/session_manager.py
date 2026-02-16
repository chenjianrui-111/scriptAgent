"""
会话管理器 - 管理会话生命周期

存储后端抽象:
  - MemoryStore: 开发/测试环境 (默认)
  - RedisStore: 生产热数据缓存
  - SQLiteStore: 本地持久化 (无需外部依赖, 适合单机部署)

通过环境变量 SESSION_STORE 切换: memory / redis / sqlite
"""

import json
import uuid
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from script_agent.models.context import (
    SessionContext, ConversationTurn, EntityCache, SlotContext,
)
from script_agent.models.message import GeneratedScript
from script_agent.config.settings import settings

logger = logging.getLogger(__name__)


# ======================================================================
#  序列化 / 反序列化
# ======================================================================

class SessionSerializer:
    """SessionContext ↔ JSON dict 互转"""

    @staticmethod
    def to_dict(session: SessionContext) -> Dict[str, Any]:
        turns = []
        for t in session.turns:
            turns.append({
                "turn_index": t.turn_index,
                "user_message": t.user_message,
                "resolved_message": t.resolved_message,
                "assistant_message": t.assistant_message,
                "generated_script": t.generated_script,
                "timestamp": t.timestamp.isoformat(),
                "token_count": t.token_count,
                "summary": t.summary,
                "is_compressed": t.is_compressed,
                "importance_score": t.importance_score,
            })

        scripts = []
        for s in session.generated_scripts:
            scripts.append({
                "script_id": s.script_id,
                "content": s.content,
                "category": s.category,
                "scenario": s.scenario,
                "style_keywords": s.style_keywords,
                "turn_index": s.turn_index,
                "adopted": s.adopted,
                "quality_score": s.quality_score,
            })

        return {
            "session_id": session.session_id,
            "tenant_id": session.tenant_id,
            "influencer_id": session.influencer_id,
            "influencer_name": session.influencer_name,
            "category": session.category,
            "created_at": session.created_at.isoformat(),
            "turns": turns,
            "entity_cache": session.entity_cache.entities,
            "slot_context": {
                "slots": session.slot_context.slots,
                "last_intent": session.slot_context.last_intent,
            },
            "generated_scripts": scripts,
            "current_state": session.current_state,
            "state_history": session.state_history,
            "workflow_snapshot": session.workflow_snapshot,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SessionContext:
        turns = []
        for t in data.get("turns", []):
            turns.append(ConversationTurn(
                turn_index=t["turn_index"],
                user_message=t["user_message"],
                resolved_message=t.get("resolved_message", ""),
                assistant_message=t["assistant_message"],
                generated_script=t.get("generated_script"),
                timestamp=datetime.fromisoformat(t["timestamp"]),
                token_count=t.get("token_count", 0),
                summary=t.get("summary"),
                is_compressed=t.get("is_compressed", False),
                importance_score=t.get("importance_score", 0.5),
            ))

        scripts = []
        for s in data.get("generated_scripts", []):
            scripts.append(GeneratedScript(
                script_id=s["script_id"],
                content=s["content"],
                category=s.get("category", ""),
                scenario=s.get("scenario", ""),
                style_keywords=s.get("style_keywords", []),
                turn_index=s.get("turn_index", 0),
                adopted=s.get("adopted", False),
                quality_score=s.get("quality_score", 0.0),
            ))

        entity_cache = EntityCache(
            entities=data.get("entity_cache", {})
        )
        sc_data = data.get("slot_context", {})
        slot_context = SlotContext(
            slots=sc_data.get("slots", {}),
            last_intent=sc_data.get("last_intent", ""),
        )

        return SessionContext(
            session_id=data["session_id"],
            tenant_id=data.get("tenant_id", ""),
            influencer_id=data.get("influencer_id", ""),
            influencer_name=data.get("influencer_name", ""),
            category=data.get("category", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            turns=turns,
            entity_cache=entity_cache,
            slot_context=slot_context,
            generated_scripts=scripts,
            current_state=data.get("current_state", "INIT"),
            state_history=data.get("state_history", []),
            workflow_snapshot=data.get("workflow_snapshot", {}),
        )


# ======================================================================
#  存储后端抽象
# ======================================================================

class SessionStore(ABC):
    """会话存储后端抽象"""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    async def set(self, session_id: str, data: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        ...

    @abstractmethod
    async def list_all(self, tenant_id: str = "") -> List[Dict[str, Any]]:
        ...

    async def close(self) -> None:
        """关闭连接 (子类可覆盖)"""
        pass


class MemoryStore(SessionStore):
    """内存存储 — 开发/测试环境"""

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._data.get(session_id)

    async def set(self, session_id: str, data: Dict[str, Any]) -> None:
        self._data[session_id] = data

    async def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    async def list_all(self, tenant_id: str = "") -> List[Dict[str, Any]]:
        results = list(self._data.values())
        if tenant_id:
            results = [r for r in results if r.get("tenant_id") == tenant_id]
        return results


class RedisStore(SessionStore):
    """Redis 存储 — 生产热数据"""

    def __init__(self, redis_url: str, prefix: str = "session:",
                 ttl: int = 7200):
        self._prefix = prefix
        self._ttl = ttl
        self._redis = None
        self._redis_url = redis_url

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self._redis_url, decode_responses=True
                )
            except ImportError:
                raise RuntimeError(
                    "redis package required: pip install 'script-agent[storage]'"
                )
        return self._redis

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        r = await self._get_redis()
        raw = await r.get(self._key(session_id))
        if raw:
            return json.loads(raw)
        return None

    async def set(self, session_id: str, data: Dict[str, Any]) -> None:
        r = await self._get_redis()
        await r.set(self._key(session_id), json.dumps(data, ensure_ascii=False),
                    ex=self._ttl)

    async def delete(self, session_id: str) -> None:
        r = await self._get_redis()
        await r.delete(self._key(session_id))

    async def list_all(self, tenant_id: str = "") -> List[Dict[str, Any]]:
        r = await self._get_redis()
        keys = []
        async for key in r.scan_iter(match=f"{self._prefix}*"):
            keys.append(key)
        results = []
        for key in keys:
            raw = await r.get(key)
            if raw:
                data = json.loads(raw)
                if not tenant_id or data.get("tenant_id") == tenant_id:
                    results.append(data)
        return results

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None


class SQLiteStore(SessionStore):
    """SQLite 存储 — 本地持久化, 无需外部依赖"""

    def __init__(self, db_path: str = "sessions.db"):
        self._db_path = db_path
        self._db = None

    async def _get_db(self):
        if self._db is None:
            try:
                import aiosqlite
            except ImportError:
                raise RuntimeError(
                    "aiosqlite package required: pip install 'script-agent[storage]'"
                )
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    tenant_id TEXT DEFAULT '',
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_tenant
                ON sessions(tenant_id)
            """)
            await self._db.commit()
        return self._db

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        db = await self._get_db()
        async with db.execute(
            "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    async def set(self, session_id: str, data: Dict[str, Any]) -> None:
        db = await self._get_db()
        now = datetime.now().isoformat()
        await db.execute("""
            INSERT INTO sessions (session_id, tenant_id, data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at
        """, (
            session_id,
            data.get("tenant_id", ""),
            json.dumps(data, ensure_ascii=False),
            data.get("created_at", now),
            now,
        ))
        await db.commit()

    async def delete(self, session_id: str) -> None:
        db = await self._get_db()
        await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await db.commit()

    async def list_all(self, tenant_id: str = "") -> List[Dict[str, Any]]:
        db = await self._get_db()
        if tenant_id:
            sql = "SELECT data FROM sessions WHERE tenant_id = ? ORDER BY created_at DESC"
            params = (tenant_id,)
        else:
            sql = "SELECT data FROM sessions ORDER BY created_at DESC"
            params = ()
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            return [json.loads(row[0]) for row in rows]

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None


# ======================================================================
#  SessionManager
# ======================================================================

def _create_store() -> SessionStore:
    """根据环境变量创建存储后端"""
    import os
    store_type = os.getenv("SESSION_STORE", "memory").lower()
    if store_type == "redis":
        return RedisStore(redis_url=settings.cache.redis_url)
    elif store_type == "sqlite":
        db_path = os.getenv("SESSION_DB_PATH", "sessions.db")
        return SQLiteStore(db_path=db_path)
    else:
        return MemoryStore()


class SessionManager:
    """
    会话管理器

    通过 SESSION_STORE 环境变量切换后端:
      - memory (默认): 内存存储, 开发/测试环境
      - redis: Redis 缓存, 生产环境
      - sqlite: SQLite 持久化, 本地部署
    """

    def __init__(self, store: Optional[SessionStore] = None):
        self._store = store or _create_store()
        self._serializer = SessionSerializer()
        logger.info(f"SessionManager using store: {type(self._store).__name__}")

    async def create(self, tenant_id: str = "",
                     influencer_id: str = "",
                     influencer_name: str = "",
                     category: str = "") -> SessionContext:
        """创建新会话"""
        session = SessionContext(
            session_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            influencer_id=influencer_id,
            influencer_name=influencer_name,
            category=category,
            created_at=datetime.now(),
        )
        await self._store.set(
            session.session_id,
            self._serializer.to_dict(session),
        )
        logger.info(f"Session created: {session.session_id}")
        return session

    async def load(self, session_id: str) -> Optional[SessionContext]:
        """加载会话"""
        data = await self._store.get(session_id)
        if data is None:
            logger.warning(f"Session not found: {session_id}")
            return None
        return self._serializer.from_dict(data)

    async def save(self, session: SessionContext):
        """保存会话"""
        await self._store.set(
            session.session_id,
            self._serializer.to_dict(session),
        )

    async def delete(self, session_id: str):
        """删除会话"""
        await self._store.delete(session_id)

    async def list_sessions(self, tenant_id: str = "") -> list:
        """列出会话"""
        all_data = await self._store.list_all(tenant_id)
        return [
            {
                "session_id": d["session_id"],
                "tenant_id": d.get("tenant_id", ""),
                "influencer_name": d.get("influencer_name", ""),
                "turn_count": len(d.get("turns", [])),
                "created_at": d.get("created_at", ""),
            }
            for d in all_data
        ]

    async def close(self):
        """关闭存储连接"""
        await self._store.close()
