"""
长期记忆检索服务（Embedding + Vector Search）

支持:
  - memory 向量库（内存）
  - elasticsearch 向量检索（可选）
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from script_agent.config.settings import settings
from script_agent.models.context import InfluencerProfile, ProductProfile, SessionContext
from script_agent.models.message import GeneratedScript, IntentResult

logger = logging.getLogger(__name__)


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return vec
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    return float(sum(a[i] * b[i] for i in range(length)))


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        ...


class HashEmbeddingProvider(EmbeddingProvider):
    """无外部依赖的哈希向量化实现，适合作为默认回退。"""

    _TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+")

    def __init__(self, dim: int = 256):
        self._dim = max(64, dim)

    def _tokenize(self, text: str) -> List[str]:
        tokens = self._TOKEN_PATTERN.findall((text or "").lower())
        if not tokens:
            return ["_empty_"]
        return tokens

    async def embed(self, text: str) -> List[float]:
        vector = [0.0] * self._dim
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self._dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[idx] += sign
        return _normalize(vector)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """可选：sentence-transformers 向量化。"""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(self, text: str) -> List[float]:
        model = self._get_model()
        values = model.encode([text or ""], normalize_embeddings=True)[0]
        return [float(v) for v in values]


class VectorStore(ABC):
    @abstractmethod
    async def upsert(
        self,
        memory_id: str,
        text: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        ...

    async def close(self) -> None:
        return None

    async def stats(self) -> Dict[str, Any]:
        return {}


class MemoryVectorStore(VectorStore):
    def __init__(self):
        self._records: Dict[str, Dict[str, Any]] = {}

    async def upsert(
        self,
        memory_id: str,
        text: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        self._records[memory_id] = {
            "memory_id": memory_id,
            "text": text,
            "vector": list(vector),
            "metadata": dict(metadata),
        }

    async def search(
        self,
        query_vector: List[float],
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for row in self._records.values():
            metadata = row.get("metadata", {})
            if any(metadata.get(k) != v for k, v in filters.items() if v):
                continue
            score = _cosine(query_vector, row.get("vector", []))
            candidates.append(
                {
                    "memory_id": row["memory_id"],
                    "text": row["text"],
                    "score": score,
                    "metadata": metadata,
                }
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[: max(1, top_k)]

    async def stats(self) -> Dict[str, Any]:
        return {"backend": "memory", "records": len(self._records)}


class ElasticsearchVectorStore(VectorStore):
    """Elasticsearch dense_vector 检索。"""

    def __init__(
        self,
        url: str,
        index: str,
        dim: int,
        timeout_seconds: int,
        username: str = "",
        password: str = "",
    ):
        self._url = url
        self._index = index
        self._dim = dim
        self._timeout_seconds = timeout_seconds
        self._username = username
        self._password = password
        self._client = None
        self._initialized = False

    async def _get_client(self):
        if self._client is None:
            from elasticsearch import AsyncElasticsearch

            kwargs: Dict[str, Any] = {"request_timeout": self._timeout_seconds}
            if self._username:
                kwargs["basic_auth"] = (self._username, self._password)
            self._client = AsyncElasticsearch(self._url, **kwargs)
        if not self._initialized:
            await self._ensure_index()
            self._initialized = True
        return self._client

    async def _ensure_index(self) -> None:
        client = self._client
        assert client is not None
        exists = await client.indices.exists(index=self._index)
        if exists:
            return
        await client.indices.create(
            index=self._index,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "vector": {"type": "dense_vector", "dims": self._dim},
                    "created_at": {"type": "date"},
                    "tenant_id": {"type": "keyword"},
                    "influencer_id": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "product_name": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        )

    async def upsert(
        self,
        memory_id: str,
        text: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        client = await self._get_client()
        doc = {
            "text": text,
            "vector": vector,
            "created_at": metadata.get("created_at", int(time.time() * 1000)),
            "tenant_id": metadata.get("tenant_id", ""),
            "influencer_id": metadata.get("influencer_id", ""),
            "category": metadata.get("category", ""),
            "product_name": metadata.get("product_name", ""),
            "metadata": metadata,
        }
        await client.index(index=self._index, id=memory_id, document=doc, refresh=False)

    async def search(
        self,
        query_vector: List[float],
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        client = await self._get_client()
        must_filters = []
        for key, value in filters.items():
            if value:
                must_filters.append({"term": {key: value}})
        body = {
            "size": max(1, top_k),
            "query": {
                "script_score": {
                    "query": {"bool": {"filter": must_filters}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                }
            },
        }
        resp = await client.search(index=self._index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        result = []
        for hit in hits:
            source = hit.get("_source", {})
            result.append(
                {
                    "memory_id": hit.get("_id", ""),
                    "text": source.get("text", ""),
                    "score": float(hit.get("_score", 0.0)) - 1.0,
                    "metadata": source.get("metadata", {}),
                }
            )
        return result

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def stats(self) -> Dict[str, Any]:
        return {"backend": "elasticsearch", "index": self._index}


def create_embedding_provider() -> EmbeddingProvider:
    cfg = settings.longterm_memory
    if cfg.embedding_backend == "sentence_transformers":
        try:
            return SentenceTransformerEmbeddingProvider(cfg.embedding_model)
        except Exception as exc:
            logger.warning(
                "init sentence-transformers embedding failed, fallback hash: %s", exc
            )
    return HashEmbeddingProvider(dim=cfg.embedding_dim)


def create_vector_store() -> VectorStore:
    cfg = settings.longterm_memory
    if cfg.backend == "elasticsearch":
        try:
            from elasticsearch import AsyncElasticsearch  # type: ignore # noqa: F401

            return ElasticsearchVectorStore(
                url=cfg.es_url,
                index=cfg.es_index,
                dim=cfg.embedding_dim,
                timeout_seconds=cfg.es_timeout_seconds,
                username=cfg.es_username,
                password=cfg.es_password,
            )
        except Exception as exc:
            logger.warning(
                "init ElasticsearchVectorStore failed, fallback memory: %s", exc
            )
    return MemoryVectorStore()


@dataclass
class RecallQuery:
    text: str
    tenant_id: str = ""
    influencer_id: str = ""
    category: str = ""
    product_name: str = ""
    top_k: Optional[int] = None


class LongTermMemoryRetriever:
    """长期记忆检索与写回管理器。"""

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ):
        self._cfg = settings.longterm_memory
        self._store = store or create_vector_store()
        self._embedder = embedder or create_embedding_provider()

    async def recall(self, query: RecallQuery) -> List[Dict[str, Any]]:
        if not self._cfg.enabled:
            return []

        query_vector = await self._embedder.embed(query.text)
        filters = {
            "tenant_id": query.tenant_id,
            "influencer_id": query.influencer_id,
            "category": query.category,
            "product_name": query.product_name,
        }
        top_k = query.top_k or self._cfg.top_k
        rows = await self._store.search(query_vector, filters, top_k)
        min_score = self._cfg.min_similarity
        return [r for r in rows if float(r.get("score", 0.0)) >= min_score]

    async def remember_script(
        self,
        session: SessionContext,
        intent: Optional[IntentResult],
        profile: Optional[InfluencerProfile],
        product: Optional[ProductProfile],
        script: GeneratedScript,
        query: str,
    ) -> Optional[str]:
        if not self._cfg.enabled or not self._cfg.write_back_enabled:
            return None
        content = (script.content or "").strip()
        if not content:
            return None

        memory_id = str(uuid.uuid4())
        text = self._build_memory_text(
            query=query,
            content=content,
            product=product,
            style_tone=(profile.style.tone if profile else ""),
        )
        vector = await self._embedder.embed(text)
        metadata = {
            "tenant_id": session.tenant_id,
            "session_id": session.session_id,
            "influencer_id": profile.influencer_id if profile else session.influencer_id,
            "category": script.category or (profile.category if profile else ""),
            "product_name": product.name if product else "",
            "scenario": script.scenario,
            "intent": intent.intent if intent else "",
            "trace_id": "",
            "created_at": int(time.time() * 1000),
        }
        await self._store.upsert(memory_id, text, vector, metadata)
        return memory_id

    def _build_memory_text(
        self,
        query: str,
        content: str,
        product: Optional[ProductProfile],
        style_tone: str,
    ) -> str:
        product_text = ""
        if product:
            product_text = (
                f"商品:{product.name}; 卖点:{','.join(product.selling_points[:5])}; "
                f"特征:{','.join(product.features[:5])}; "
            )
        return (
            f"query:{query}\n"
            f"{product_text}"
            f"风格:{style_tone}\n"
            f"script:{content}"
        )

    async def close(self) -> None:
        await self._store.close()

    async def stats(self) -> Dict[str, Any]:
        base = await self._store.stats()
        base["enabled"] = self._cfg.enabled
        base["embedding_backend"] = self._cfg.embedding_backend
        return base
