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
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from script_agent.config.settings import settings
from script_agent.models.context import InfluencerProfile, ProductProfile, SessionContext
from script_agent.models.message import GeneratedScript, IntentResult

logger = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+")


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


def _tokenize(text: str) -> List[str]:
    tokens = _TOKEN_PATTERN.findall((text or "").lower())
    return tokens if tokens else ["_empty_"]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _normalize_scores(rows: List[Dict[str, Any]], score_key: str) -> Dict[str, float]:
    if not rows:
        return {}
    values = [float(r.get(score_key, 0.0)) for r in rows]
    high = max(values)
    low = min(values)
    if abs(high - low) < 1e-9:
        return {str(r.get("memory_id", "")): 1.0 for r in rows}
    return {
        str(r.get("memory_id", "")): (
            (float(r.get(score_key, 0.0)) - low) / (high - low)
        )
        for r in rows
    }


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        ...


class HashEmbeddingProvider(EmbeddingProvider):
    """无外部依赖的哈希向量化实现，适合作为默认回退。"""

    def __init__(self, dim: int = 256):
        self._dim = max(64, dim)

    def _tokenize(self, text: str) -> List[str]:
        return _tokenize(text)

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
        query_text: str = "",
    ) -> List[Dict[str, Any]]:
        ...

    async def search_sparse(
        self,
        query_text: str,
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        return []

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
        query_text: str = "",
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

    async def search_sparse(
        self,
        query_text: str,
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        query_tokens = _tokenize(query_text)
        docs: List[Tuple[Dict[str, Any], List[str]]] = []
        for row in self._records.values():
            metadata = row.get("metadata", {})
            if any(metadata.get(k) != v for k, v in filters.items() if v):
                continue
            text_tokens = _tokenize(row.get("text", ""))
            docs.append((row, text_tokens))

        if not docs:
            return []

        doc_count = len(docs)
        avg_len = sum(len(tks) for _, tks in docs) / max(1, doc_count)
        query_set = set(query_tokens)
        df = Counter()
        for _, tokens in docs:
            for token in set(tokens):
                if token in query_set:
                    df[token] += 1

        candidates: List[Dict[str, Any]] = []
        k1 = 1.2
        b = 0.75
        for row, tokens in docs:
            tf = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq <= 0:
                    continue
                denom = freq + k1 * (1 - b + b * doc_len / max(1e-6, avg_len))
                idf = math.log(1 + (doc_count - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
                score += idf * ((freq * (k1 + 1)) / max(1e-9, denom))

            candidates.append(
                {
                    "memory_id": row["memory_id"],
                    "text": row["text"],
                    "score": score,
                    "metadata": row.get("metadata", {}),
                }
            )

        if not candidates:
            return []
        normalized = _normalize_scores(candidates, "score")
        for row in candidates:
            row["score"] = normalized.get(str(row.get("memory_id", "")), 0.0)
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
        query_text: str = "",
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

    async def search_sparse(
        self,
        query_text: str,
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
                "bool": {
                    "filter": must_filters,
                    "must": [
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": [
                                    "text^3",
                                    "product_name^2",
                                    "category",
                                    "metadata.scenario",
                                    "metadata.intent",
                                ],
                                "type": "best_fields",
                            }
                        }
                    ],
                }
            },
        }
        resp = await client.search(index=self._index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        result: List[Dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source", {})
            result.append(
                {
                    "memory_id": hit.get("_id", ""),
                    "text": source.get("text", ""),
                    "score": float(hit.get("_score", 0.0)),
                    "metadata": source.get("metadata", {}),
                }
            )
        normalized = _normalize_scores(result, "score")
        for row in result:
            row["score"] = normalized.get(str(row.get("memory_id", "")), 0.0)
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
    intent: str = ""
    scenario: str = ""
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
        top_k = self._resolve_top_k(query)
        candidate_k = max(top_k, top_k * max(1, self._cfg.hybrid_candidate_multiplier))

        dense_rows = await self._store.search(
            query_vector=query_vector,
            filters=filters,
            top_k=candidate_k,
            query_text=query.text,
        )
        rows = dense_rows
        if self._cfg.hybrid_enabled:
            try:
                sparse_rows = await self._store.search_sparse(
                    query_text=query.text,
                    filters=filters,
                    top_k=candidate_k,
                )
                rows = self._hybrid_fuse(dense_rows, sparse_rows, query)
            except Exception as exc:
                logger.warning("long-term sparse recall failed, fallback dense: %s", exc)
                rows = dense_rows

        if self._cfg.rerank_enabled:
            rows = self._rerank(rows, query)

        rows = rows[:top_k]
        min_score = self._cfg.min_similarity
        return [r for r in rows if float(r.get("score", 0.0)) >= min_score]

    def _resolve_top_k(self, query: RecallQuery) -> int:
        base = query.top_k if query.top_k and query.top_k > 0 else self._cfg.top_k
        if not self._cfg.adaptive_top_k_enabled:
            return max(self._cfg.top_k_min, min(base, self._cfg.top_k_max))

        scenario_text = (query.scenario or query.text or "").lower()
        intent = (query.intent or "").lower()
        delta = 0

        # 场景越复杂，召回窗口越大
        if any(k in scenario_text for k in ("开场", "问候", "破冰")):
            delta -= 1
        if any(k in scenario_text for k in ("产品介绍", "讲解", "卖点")):
            delta += 1
        if any(k in scenario_text for k in ("促销", "活动", "折扣", "大促")):
            delta += 2
        if any(k in scenario_text for k in ("优化", "改写", "仿写", "续写")):
            delta += 1
        if query.product_name:
            delta += 1
        if intent in {"script_modification", "script_optimization"}:
            delta += 1

        resolved = base + delta
        return max(self._cfg.top_k_min, min(resolved, self._cfg.top_k_max))

    def _hybrid_fuse(
        self,
        dense_rows: List[Dict[str, Any]],
        sparse_rows: List[Dict[str, Any]],
        query: RecallQuery,
    ) -> List[Dict[str, Any]]:
        dense_norm = {
            str(r.get("memory_id", "")): (float(r.get("score", 0.0)) + 1.0) / 2.0
            for r in dense_rows
        }
        sparse_norm = {
            str(r.get("memory_id", "")): float(r.get("score", 0.0))
            for r in sparse_rows
        }
        dense_rank = {str(r.get("memory_id", "")): idx + 1 for idx, r in enumerate(dense_rows)}
        sparse_rank = {str(r.get("memory_id", "")): idx + 1 for idx, r in enumerate(sparse_rows)}

        merged: Dict[str, Dict[str, Any]] = {}
        for row in dense_rows + sparse_rows:
            memory_id = str(row.get("memory_id", ""))
            if not memory_id:
                continue
            if memory_id not in merged:
                merged[memory_id] = {
                    "memory_id": memory_id,
                    "text": row.get("text", ""),
                    "metadata": dict(row.get("metadata", {})),
                    "dense_score": dense_norm.get(memory_id, 0.0),
                    "sparse_score": sparse_norm.get(memory_id, 0.0),
                }
            else:
                if not merged[memory_id].get("text"):
                    merged[memory_id]["text"] = row.get("text", "")
                if not merged[memory_id].get("metadata"):
                    merged[memory_id]["metadata"] = dict(row.get("metadata", {}))

        fused_rows: List[Dict[str, Any]] = []
        rrf_k = 60.0
        wd = max(0.0, self._cfg.hybrid_dense_weight)
        ws = max(0.0, self._cfg.hybrid_sparse_weight)
        total = max(1e-9, wd + ws)
        wd, ws = wd / total, ws / total
        for memory_id, row in merged.items():
            dense_score = row.get("dense_score", 0.0)
            sparse_score = row.get("sparse_score", 0.0)
            rank_boost = 0.0
            if memory_id in dense_rank:
                rank_boost += wd / (rrf_k + dense_rank[memory_id])
            if memory_id in sparse_rank:
                rank_boost += ws / (rrf_k + sparse_rank[memory_id])
            fused = (wd * dense_score) + (ws * sparse_score) + rank_boost
            row["score"] = max(0.0, min(1.0, fused))
            fused_rows.append(row)

        fused_rows.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return fused_rows

    def _rerank(
        self,
        rows: List[Dict[str, Any]],
        query: RecallQuery,
    ) -> List[Dict[str, Any]]:
        if not rows:
            return rows
        window = max(1, self._cfg.rerank_window)
        limited = rows[:window]
        query_tokens = _tokenize(query.text)
        reranked: List[Dict[str, Any]] = []
        for row in limited:
            metadata = row.get("metadata", {})
            text_tokens = _tokenize(str(row.get("text", "")))
            lexical = _jaccard(query_tokens, text_tokens)
            boost = 0.0
            if query.product_name and metadata.get("product_name") == query.product_name:
                boost += 0.12
            if query.category and metadata.get("category") == query.category:
                boost += 0.06
            if query.influencer_id and metadata.get("influencer_id") == query.influencer_id:
                boost += 0.06
            if query.intent and metadata.get("intent") == query.intent:
                boost += 0.04
            if query.scenario and metadata.get("scenario") == query.scenario:
                boost += 0.03

            base_score = float(row.get("score", 0.0))
            row["score"] = max(0.0, min(1.0, base_score * 0.72 + lexical * 0.28 + boost))
            reranked.append(row)

        reranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        if len(rows) <= window:
            return reranked
        return reranked + rows[window:]

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
