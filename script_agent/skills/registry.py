"""
Skill 注册中心 — 管理所有 Skill 的注册、发现与路由
"""

import logging
from typing import Dict, Iterable, List, Optional, Tuple

from script_agent.skills.base import BaseSkill

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
            }
            for s in self._skills.values()
        ]

    def iter_skills(self) -> Iterable[BaseSkill]:
        """遍历所有已注册 Skill"""
        return self._skills.values()
