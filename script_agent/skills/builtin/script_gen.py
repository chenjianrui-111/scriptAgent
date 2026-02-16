"""
话术生成 Skill — 将现有核心生成流程包装为 Skill
"""

from typing import Any, Dict

from script_agent.skills.base import BaseSkill, SkillContext, SkillResult
from script_agent.agents.script_agent import ScriptGenerationAgent
from script_agent.agents.quality_agent import QualityCheckAgent
from script_agent.models.message import AgentMessage, GeneratedScript
from script_agent.config.settings import settings


class ScriptGenerationSkill(BaseSkill):
    """话术生成 Skill — 核心业务"""

    name = "script_generation"
    display_name = "话术生成"
    description = "根据品类、场景、达人风格生成话术"
    required_slots = ["category", "scenario"]

    def __init__(self):
        self._script_agent = ScriptGenerationAgent()
        self._quality_agent = QualityCheckAgent()

    def can_handle(self, intent: str, slots: Dict[str, Any]) -> float:
        if intent == "script_generation":
            return 1.0
        if intent in ("script_optimization",):
            return 0.6
        return 0.0

    async def execute(self, context: SkillContext) -> SkillResult:
        slots = context.intent.slots
        profile = context.profile
        session = context.session

        # 生成
        script_msg = AgentMessage(
            trace_id=context.trace_id,
            payload={"slots": slots, "profile": profile, "session": session},
            session_id=session.session_id,
        )
        script_resp = await self._script_agent(script_msg)
        script: GeneratedScript = script_resp.payload.get("script", GeneratedScript())

        # 质量校验 + 重试
        retry_count = 0
        quality_result = None
        while retry_count <= settings.quality.max_retries:
            quality_msg = AgentMessage(
                trace_id=context.trace_id,
                payload={"script": script, "profile": profile},
                session_id=session.session_id,
            )
            quality_resp = await self._quality_agent(quality_msg)
            quality_result = quality_resp.payload.get("quality_result")

            if quality_result and quality_result.passed:
                break

            retry_count += 1
            if retry_count <= settings.quality.max_retries:
                script_msg.payload["feedback"] = (
                    quality_result.suggestions if quality_result else []
                )
                script_resp = await self._script_agent(script_msg)
                script = script_resp.payload.get("script", script)

        return SkillResult(
            success=quality_result.passed if quality_result else False,
            script=script,
            quality_result=quality_result,
        )
