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
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from script_agent.models.context import (
    SessionContext, ConversationTurn, EntityCache, SlotContext,
)
from script_agent.models.message import GeneratedScript
from script_agent.config.settings import settings
from script_agent.observability import metrics as obs

logger = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+")


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
            "owner_user_id": session.owner_user_id,
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
            owner_user_id=data.get("owner_user_id", ""),
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
                     owner_user_id: str = "",
                     influencer_id: str = "",
                     influencer_name: str = "",
                     category: str = "") -> SessionContext:
        """创建新会话"""
        session = SessionContext(
            session_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            influencer_id=influencer_id,
            influencer_name=influencer_name,
            category=category,
            created_at=datetime.now(),
        )
        await self._store.set(
            session.session_id,
            self._serializer.to_dict(session),
        )
        obs.mark_session_created(session.session_id)
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
        self._apply_retention_policy(session)
        await self._store.set(
            session.session_id,
            self._serializer.to_dict(session),
        )

    async def delete(self, session_id: str):
        """删除会话"""
        await self._store.delete(session_id)
        obs.mark_session_deleted(session_id)

    async def list_sessions(self, tenant_id: str = "", owner_user_id: str = "") -> list:
        """列出会话"""
        all_data = await self._store.list_all(tenant_id)
        if owner_user_id:
            all_data = [
                d for d in all_data
                if str(d.get("owner_user_id", "")).strip() == owner_user_id
            ]
        return [
            {
                "session_id": d["session_id"],
                "tenant_id": d.get("tenant_id", ""),
                "owner_user_id": d.get("owner_user_id", ""),
                "influencer_name": d.get("influencer_name", ""),
                "turn_count": len(d.get("turns", [])),
                "created_at": d.get("created_at", ""),
            }
            for d in all_data
        ]

    async def close(self):
        """关闭存储连接"""
        await self._store.close()

    def _apply_retention_policy(self, session: SessionContext) -> None:
        """会话记忆裁剪 + 压缩（按任务相关性动态保留）"""
        cfg = settings.context
        relevance_scores = self._score_turn_relevance(session)
        if cfg.compress_history_on_save:
            self._compress_old_turns(
                session,
                keep_recent=max(1, cfg.zone_a_turns),
                relevance_scores=relevance_scores,
            )
        self._trim_turns(
            session,
            max_turns=cfg.max_turns_persisted,
            relevance_scores=relevance_scores,
        )
        self._trim_scripts(session, max_scripts=cfg.max_scripts_persisted)

    def _compress_old_turns(
        self,
        session: SessionContext,
        keep_recent: int,
        relevance_scores: Dict[int, float],
    ) -> None:
        max_chars = max(40, settings.context.compress_message_max_chars)
        old_turns = session.turns[:-keep_recent] if len(session.turns) > keep_recent else []
        preserve_top = max(0, settings.context.relevance_preserve_old_turns)
        preserved_indexes = {
            idx
            for idx, _ in sorted(
                (
                    (turn.turn_index, relevance_scores.get(turn.turn_index, 0.0))
                    for turn in old_turns
                ),
                key=lambda x: x[1],
                reverse=True,
            )[:preserve_top]
        }
        for turn in old_turns:
            if turn.is_compressed:
                continue
            if turn.turn_index in preserved_indexes:
                continue
            user = self._clip_text(turn.user_message, max_chars)
            assistant = self._clip_text(turn.assistant_message, max_chars)

            if turn.intent:
                key_slots = ",".join(
                    f"{k}={v}"
                    for k, v in turn.intent.slots.items()
                    if v and not str(k).startswith("_")
                )
                summary = f"intent={turn.intent.intent};slots={key_slots}"
            else:
                summary = ""

            turn.summary = summary
            if summary:
                user = f"[{summary}] {user}"
            if turn.generated_script:
                assistant = f"[已生成话术] {assistant}"

            turn.user_message = user
            turn.assistant_message = assistant
            turn.is_compressed = True
            turn.token_count = int((len(user) + len(assistant)) / 1.5)

    def _trim_turns(
        self,
        session: SessionContext,
        max_turns: int,
        relevance_scores: Dict[int, float],
    ) -> None:
        if max_turns <= 0 or len(session.turns) <= max_turns:
            return

        cfg = settings.context
        keep_recent = max(1, cfg.zone_a_turns)
        keep_recent = min(keep_recent, max_turns)
        recent = session.turns[-keep_recent:]
        old_candidates = session.turns[:-keep_recent]
        remaining_slots = max_turns - len(recent)

        kept_old: List[ConversationTurn] = []
        if remaining_slots > 0 and old_candidates:
            if cfg.relevance_trim_enabled:
                ranked = sorted(
                    old_candidates,
                    key=lambda t: relevance_scores.get(t.turn_index, 0.0),
                    reverse=True,
                )
                kept_old = ranked[:remaining_slots]
            else:
                kept_old = old_candidates[-remaining_slots:]

        kept_turns = sorted(
            kept_old + list(recent),
            key=lambda t: t.turn_index,
        )
        old_to_new = {turn.turn_index: idx for idx, turn in enumerate(kept_turns)}
        kept_old_indices = sorted(old_to_new.keys())

        session.turns = kept_turns
        for idx, turn in enumerate(session.turns):
            turn.turn_index = idx

        for script in session.generated_scripts:
            script.turn_index = self._remap_turn_index(
                script.turn_index,
                old_to_new,
                kept_old_indices,
            )

    def _trim_scripts(self, session: SessionContext, max_scripts: int) -> None:
        if max_scripts <= 0 or len(session.generated_scripts) <= max_scripts:
            return
        session.generated_scripts = session.generated_scripts[-max_scripts:]

    def _clip_text(self, text: str, max_chars: int) -> str:
        value = re.sub(r"\s+", " ", (text or "")).strip()
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3] + "..."

    def _score_turn_relevance(self, session: SessionContext) -> Dict[int, float]:
        if not session.turns:
            return {}
        query, intent_name, product_name = self._extract_retention_signals(session)
        query_tokens = self._tokenize(query)
        product_tokens = self._tokenize(product_name)
        total_turns = len(session.turns)
        cfg = settings.context

        scores: Dict[int, float] = {}
        for turn in session.turns:
            corpus = " ".join(
                [
                    turn.user_message or "",
                    turn.assistant_message or "",
                    turn.generated_script or "",
                    turn.summary or "",
                ]
            )
            text_tokens = self._tokenize(corpus)
            query_overlap = self._overlap_ratio(query_tokens, text_tokens)
            product_overlap = self._overlap_ratio(product_tokens, text_tokens)
            intent_match = 1.0 if (turn.intent and turn.intent.intent == intent_name) else 0.0
            recency = (turn.turn_index + 1) / max(1, total_turns)

            score = (
                cfg.relevance_query_weight * query_overlap
                + cfg.relevance_product_weight * product_overlap
                + cfg.relevance_intent_weight * intent_match
                + cfg.relevance_recency_weight * recency
            )
            if turn.generated_script:
                score += 0.05
            scores[turn.turn_index] = min(1.0, max(0.0, score))
        return scores

    def _extract_retention_signals(self, session: SessionContext) -> Tuple[str, str, str]:
        snapshot = session.workflow_snapshot or {}
        query = str(snapshot.get("last_query", "")).strip()
        intent_name = ""
        product_name = ""

        intent_data = snapshot.get("intent") if isinstance(snapshot, dict) else None
        if isinstance(intent_data, dict):
            intent_name = str(intent_data.get("intent", "")).strip()
            slots = intent_data.get("slots", {})
            if isinstance(slots, dict):
                product_name = str(slots.get("product_name", "")).strip()
                if not query:
                    query = str(slots.get("requirements", "")).strip()

        if not product_name:
            latest_product = session.entity_cache.get_latest("product")
            if isinstance(latest_product, dict):
                product_name = str(latest_product.get("name", "")).strip()

        if not query and session.turns:
            query = session.turns[-1].user_message

        if not intent_name and session.turns and session.turns[-1].intent:
            intent_name = session.turns[-1].intent.intent

        return query, intent_name, product_name

    def _tokenize(self, text: str) -> List[str]:
        return _TOKEN_PATTERN.findall((text or "").lower())

    def _overlap_ratio(self, a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        return len(sa & sb) / max(1, len(sa))

    def _remap_turn_index(
        self,
        old_index: int,
        old_to_new: Dict[int, int],
        kept_old_indices: List[int],
    ) -> int:
        if old_index in old_to_new:
            return old_to_new[old_index]
        if not kept_old_indices:
            return 0
        nearest = kept_old_indices[0]
        for candidate in kept_old_indices:
            if candidate <= old_index:
                nearest = candidate
            else:
                break
        return old_to_new.get(nearest, 0)
