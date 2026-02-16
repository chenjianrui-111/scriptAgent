"""
Skill 基类与上下文

Skill 是可插拔的能力模块, 将业务逻辑解耦为独立的执行单元。
每个 Skill 声明:
  - 自己能处理哪些意图 (can_handle)
  - 需要哪些槽位 (required_slots)
  - 执行逻辑 (execute)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from script_agent.models.message import IntentResult, GeneratedScript, QualityResult
from script_agent.models.context import SessionContext, InfluencerProfile

logger = logging.getLogger(__name__)


@dataclass
class SkillContext:
    """Skill 执行上下文 — 包含所有 Skill 执行所需信息"""
    intent: IntentResult
    profile: InfluencerProfile
    session: SessionContext
    trace_id: str = ""
    query: str = ""
    role: str = "user"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """Skill 执行结果"""
    success: bool
    script: Optional[GeneratedScript] = None
    quality_result: Optional[QualityResult] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


class BaseSkill(ABC):
    """
    Skill 抽象基类

    子类需实现:
      - name: 唯一标识
      - display_name: 展示名
      - description: 描述 (用于意图路由)
      - required_slots: 必需槽位列表
      - execute: 执行逻辑
    """

    name: str = ""
    display_name: str = ""
    description: str = ""
    required_slots: List[str] = []
    input_schema: Dict[str, Any] = {}

    def can_handle(self, intent: str, slots: Dict[str, Any]) -> float:
        """
        返回该 Skill 处理当前意图的置信度 0-1

        默认实现基于 name 精确匹配 intent。
        子类可覆盖以实现更复杂的匹配逻辑。
        """
        if intent == self.name:
            return 1.0
        return 0.0

    def validate_slots(self, slots: Dict[str, Any]) -> Optional[str]:
        """检查必需槽位, 返回缺失信息或 None"""
        missing = [s for s in self.required_slots if not slots.get(s)]
        if missing:
            return f"缺少必要信息: {', '.join(missing)}"
        return None

    def get_input_schema(self) -> Dict[str, Any]:
        """
        返回工具输入 schema。
        子类可通过 `input_schema` 定义严格约束。
        """
        if self.input_schema:
            return self.input_schema
        props = {name: {"type": "string", "minLength": 1} for name in self.required_slots}
        return {
            "type": "object",
            "properties": props,
            "required": list(self.required_slots),
            "additionalProperties": True,
        }

    @abstractmethod
    async def execute(self, context: SkillContext) -> SkillResult:
        """执行 Skill"""
        ...
