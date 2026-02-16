"""
Skill 注册中心 — 管理所有 Skill 的注册、发现与路由
"""

import logging
from typing import Dict, Iterable, List, Optional, Tuple

from script_agent.config.settings import settings
from script_agent.skills.base import BaseSkill
from script_agent.skills.security import (
    PromptInjectionTripwire,
    StrictJSONSchemaValidator,
    ToolPolicyEngine,
)

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    Skill 注册中心

    功能:
      - 注册/注销 Skill
      - 根据意图+槽位路由到最佳 Skill
      - 列出所有可用 Skill
    """

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}
        self._schema_validator = StrictJSONSchemaValidator(
            strict_enabled=settings.tool_security.schema_strict_enabled
        )
        self._policy_engine = ToolPolicyEngine(settings.tool_security)
        self._tripwire = PromptInjectionTripwire(settings.tool_security)

    def register(self, skill: BaseSkill):
        """注册 Skill"""
        if skill.name in self._skills:
            logger.warning(f"Skill '{skill.name}' already registered, overwriting")
        self._skills[skill.name] = skill
        logger.info(f"Skill registered: {skill.name} ({skill.display_name})")

    def unregister(self, name: str):
        """注销 Skill"""
        self._skills.pop(name, None)

    def get(self, name: str) -> Optional[BaseSkill]:
        """按名称获取 Skill"""
        return self._skills.get(name)

    def route(self, intent: str, slots: Dict) -> Optional[BaseSkill]:
        """
        根据意图路由到最佳 Skill

        Returns:
            匹配度最高的 Skill, 或 None (无匹配)
        """
        candidates: List[Tuple[BaseSkill, float]] = []
        for skill in self._skills.values():
            score = skill.can_handle(intent, slots)
            if score > 0:
                candidates.append((skill, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_skill, best_score = candidates[0]

        if best_score >= 0.5:
            logger.info(
                f"Skill routed: {intent} → {best_skill.name} "
                f"(score={best_score:.2f})"
            )
            return best_skill

        return None

    def list_skills(self) -> List[Dict]:
        """列出所有已注册 Skill"""
        return [
            {
                "name": s.name,
                "display_name": s.display_name,
                "description": s.description,
                "required_slots": s.required_slots,
                "input_schema": s.get_input_schema(),
            }
            for s in self._skills.values()
        ]

    def iter_skills(self) -> Iterable[BaseSkill]:
        """遍历所有已注册 Skill"""
        return self._skills.values()

    def preflight(
        self,
        skill: BaseSkill,
        slots: Dict,
        tenant_id: str,
        role: str,
        query: str,
    ) -> Optional[str]:
        """
        工具调用前置安全检查:
          1) required slots
          2) strict json schema
          3) tenant/role allowlist
          4) prompt injection tripwire
        """
        missing_error = skill.validate_slots(slots)
        if missing_error:
            return missing_error

        schema_error = self._schema_validator.validate(slots, skill.get_input_schema())
        if schema_error:
            return f"tool schema validation failed: {schema_error}"

        policy_decision = self._policy_engine.evaluate(
            skill_name=skill.name,
            tenant_id=tenant_id,
            role=role,
        )
        if not policy_decision.allowed:
            return f"tool policy denied: {policy_decision.reason}"

        tripwire_error = self._tripwire.inspect(query=query, slots=slots)
        if tripwire_error:
            return f"tool tripwire blocked: {tripwire_error}"
        return None
