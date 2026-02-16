"""
商品 Agent - 基于商品信息构建卖点画像，并召回长期记忆样本
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from script_agent.agents.base import BaseAgent
from script_agent.models.context import InfluencerProfile, ProductProfile, SessionContext
from script_agent.models.message import AgentMessage
from script_agent.services.long_term_memory import LongTermMemoryRetriever, RecallQuery

logger = logging.getLogger(__name__)


class ProductKnowledgeBase:
    """轻量商品卖点知识库（默认兜底）。"""

    DEFAULT_POINTS: Dict[str, Dict[str, List[str]]] = {
        "美妆": {
            "selling_points": ["成分安全", "上脸服帖", "妆效自然", "性价比高"],
            "compliance_notes": ["避免医疗功效描述", "避免绝对化承诺"],
        },
        "食品": {
            "selling_points": ["配料干净", "口感层次丰富", "场景适配强", "复购率高"],
            "compliance_notes": ["避免保健/药效描述", "避免夸大营养功能"],
        },
        "服饰": {
            "selling_points": ["版型友好", "面料舒适", "通勤百搭", "耐穿耐洗"],
            "compliance_notes": ["避免绝对化描述", "尺码建议需客观"],
        },
        "数码": {
            "selling_points": ["性能稳定", "使用门槛低", "核心参数清晰", "售后有保障"],
            "compliance_notes": ["参数描述需真实", "避免误导性对比"],
        },
    }

    def build_defaults(self, category: str) -> Tuple[List[str], List[str]]:
        data = self.DEFAULT_POINTS.get(category, {})
        return list(data.get("selling_points", [])), list(data.get("compliance_notes", []))


class ProductAgent(BaseAgent):
    def __init__(self, memory: Optional[LongTermMemoryRetriever] = None):
        super().__init__(name="product")
        self._knowledge = ProductKnowledgeBase()
        self._memory = memory or LongTermMemoryRetriever()

    async def process(self, message: AgentMessage) -> AgentMessage:
        slots = dict(message.payload.get("slots", {}))
        profile: InfluencerProfile = message.payload.get("profile", InfluencerProfile())
        session: SessionContext = message.payload.get("session", SessionContext())
        query: str = message.payload.get("query", "")

        product = self._build_product_profile(slots, profile)
        memory_hits = await self._recall_memories(
            query=query,
            session=session,
            profile=profile,
            product=product,
        )

        return message.create_response(
            payload={
                "product": product,
                "memory_hits": memory_hits,
            },
            source=self.name,
        )

    async def fetch(
        self,
        slots: Dict[str, Any],
        profile: InfluencerProfile,
        session: SessionContext,
        query: str = "",
    ) -> Tuple[ProductProfile, List[Dict[str, Any]]]:
        msg = AgentMessage(
            payload={
                "slots": slots,
                "profile": profile,
                "session": session,
                "query": query,
            },
            session_id=session.session_id,
            tenant_id=session.tenant_id,
        )
        resp = await self.process(msg)
        return (
            resp.payload.get("product", ProductProfile()),
            resp.payload.get("memory_hits", []),
        )

    def _build_product_profile(
        self,
        slots: Dict[str, Any],
        profile: InfluencerProfile,
    ) -> ProductProfile:
        category = slots.get("category") or profile.category or "通用"
        name = (slots.get("product_name") or "").strip()
        if not name and slots.get("requirements"):
            name = self._guess_product_name(slots["requirements"])
        if not name:
            name = f"{category}商品"

        features = self._to_list(slots.get("product_features"))
        selling_points = self._to_list(slots.get("selling_points"))
        default_points, default_notes = self._knowledge.build_defaults(category)

        merged_points = list(dict.fromkeys(selling_points + features + default_points))
        return ProductProfile(
            product_id=slots.get("product_id", ""),
            name=name,
            category=category,
            brand=slots.get("brand", ""),
            price_range=slots.get("price_range", ""),
            features=features,
            selling_points=merged_points[:8],
            target_audience=slots.get("target_audience", profile.audience_age_range),
            compliance_notes=default_notes,
        )

    async def _recall_memories(
        self,
        query: str,
        session: SessionContext,
        profile: InfluencerProfile,
        product: ProductProfile,
    ) -> List[Dict[str, Any]]:
        recall_query = RecallQuery(
            text=self._build_recall_query(query, profile, product),
            tenant_id=session.tenant_id,
            influencer_id=profile.influencer_id or session.influencer_id,
            category=product.category,
            product_name=product.name,
        )
        try:
            return await self._memory.recall(recall_query)
        except Exception as exc:
            logger.warning("product memory recall failed: %s", exc)
            return []

    def _build_recall_query(
        self,
        query: str,
        profile: InfluencerProfile,
        product: ProductProfile,
    ) -> str:
        return (
            f"query:{query};"
            f"达人风格:{profile.style.tone};"
            f"商品:{product.name};"
            f"卖点:{','.join(product.selling_points[:5])};"
            f"场景偏好:{product.category}"
        )

    def _guess_product_name(self, text: str) -> str:
        # 简单规则: 抓取“这款XXX/这个XXX/XXX这款”中的核心名词
        patterns = [
            re.compile(r"(?:这款|这个|该)([\u4e00-\u9fffA-Za-z0-9]{2,24})"),
            re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,24})(?:这款|这个)"),
        ]
        for pattern in patterns:
            m = pattern.search(text)
            if m:
                return m.group(1).strip()
        return ""

    def _to_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [s.strip() for s in re.split(r"[，,、;；/\n]", value) if s.strip()]
        return []
