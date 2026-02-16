"""
达人画像Agent - 多级缓存 + 实时构建 + 特征向量化

缓存层级:
  L1: 本地内存 (热门达人) ~1ms
  L2: Redis ~5ms
  L3: 实时构建 (LLM总结) ~200ms
"""

import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from script_agent.agents.base import BaseAgent
from script_agent.models.message import AgentMessage
from script_agent.models.context import InfluencerProfile, StyleProfile
from script_agent.services.llm_client import LLMServiceClient
from script_agent.config.settings import settings

logger = logging.getLogger(__name__)


class LRULocalCache:
    """L1: 本地内存LRU缓存"""

    def __init__(self, maxsize: int = 200):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


class ProfileCache:
    """
    多级画像缓存
    L1(本地内存) → L2(Redis) → L3(实时构建)
    """

    def __init__(self):
        self.local_cache = LRULocalCache(maxsize=200)
        # L2 Redis (实际项目中注入redis client)
        self._redis_mock: Dict[str, Any] = {}

    async def get_profile(self, target_id: str) -> Optional[InfluencerProfile]:
        # L1: 本地内存 (~1ms)
        cached = self.local_cache.get(f"profile:{target_id}")
        if cached:
            logger.debug(f"Profile cache L1 HIT: {target_id}")
            return cached

        # L2: Redis (~5ms)
        redis_data = self._redis_mock.get(f"profile:{target_id}")
        if redis_data:
            logger.debug(f"Profile cache L2 HIT: {target_id}")
            self.local_cache.set(f"profile:{target_id}", redis_data)
            return redis_data

        return None

    async def set_profile(self, target_id: str, profile: InfluencerProfile):
        self.local_cache.set(f"profile:{target_id}", profile)
        self._redis_mock[f"profile:{target_id}"] = profile


class ProfileBuilder:
    """
    实时画像构建 - 当缓存未命中时, 拉取达人历史内容通过LLM总结风格
    """

    def __init__(self):
        self.llm = LLMServiceClient()

    async def build(self, target_id: str = "",
                    target_name: str = "",
                    category: str = "") -> InfluencerProfile:
        """构建达人画像"""
        # 实际项目: 从HBase/数据库拉取达人历史内容
        # 这里模拟构建
        style = await self._extract_style(target_name, category)

        return InfluencerProfile(
            influencer_id=target_id or f"inf_{target_name}",
            name=target_name,
            category=category,
            style=style,
        )

    async def build_generic(self, category: str) -> InfluencerProfile:
        """构建品类通用画像 (无指定达人时使用)"""
        style_map = {
            "美妆": StyleProfile(
                tone="活泼亲和", formality_level=0.3,
                catchphrases=["姐妹们", "宝子们", "绝了"],
                avg_sentence_length=15, humor_level=0.5,
            ),
            "食品": StyleProfile(
                tone="生活化", formality_level=0.4,
                catchphrases=["家人们", "好吃到哭"],
                avg_sentence_length=18, humor_level=0.4,
            ),
            "服饰": StyleProfile(
                tone="时尚专业", formality_level=0.5,
                catchphrases=["小仙女们", "质感拉满"],
                avg_sentence_length=20, humor_level=0.3,
            ),
        }
        return InfluencerProfile(
            influencer_id=f"generic_{category}",
            name=f"{category}通用达人",
            category=category,
            style=style_map.get(category, StyleProfile(tone="通用")),
        )

    async def _extract_style(self, name: str, category: str) -> StyleProfile:
        """通过LLM提取风格特征 (模拟)"""
        # 实际项目: LLM分析达人历史优质内容
        return StyleProfile(
            tone="活泼", formality_level=0.3,
            catchphrases=["宝子们"], humor_level=0.4,
            confidence=0.6,
        )


class ProfileAgent(BaseAgent):
    """
    达人画像Agent
    输入: slots (target_id / target_name / category)
    输出: InfluencerProfile
    """

    def __init__(self):
        super().__init__(name="profile")
        self.cache = ProfileCache()
        self.builder = ProfileBuilder()

    async def process(self, message: AgentMessage) -> AgentMessage:
        slots = message.payload.get("slots", {})
        target_id = slots.get("target_id", "")
        target_name = slots.get("target_name", "")
        category = slots.get("category", "通用")

        profile: Optional[InfluencerProfile] = None

        # 路径1: 有target_id → 从缓存/数据库获取
        if target_id or target_name:
            lookup_key = target_id or target_name
            profile = await self.cache.get_profile(lookup_key)
            if not profile:
                # 缓存未命中 → 实时构建
                logger.info(f"Profile cache MISS, building for: {lookup_key}")
                profile = await self.builder.build(
                    target_id, target_name, category
                )
                await self.cache.set_profile(lookup_key, profile)
        else:
            # 路径2: 无target → 品类通用画像
            profile = await self.builder.build_generic(category)

        return message.create_response(
            payload={"profile": profile},
            source=self.name,
        )

    async def fetch(self, slots: Dict[str, Any]) -> InfluencerProfile:
        """直接调用接口 (供Orchestrator并行调用)"""
        msg = AgentMessage(payload={"slots": slots})
        result = await self.process(msg)
        return result.payload.get("profile", InfluencerProfile())
