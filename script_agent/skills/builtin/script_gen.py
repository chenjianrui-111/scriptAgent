"""
话术生成 Skill — 将现有核心生成流程包装为 Skill
"""

from typing import Any, Dict

from script_agent.skills.base import BaseSkill, SkillContext, SkillResult
from script_agent.agents.script_agent import ScriptGenerationAgent
from script_agent.agents.quality_agent import QualityCheckAgent
from script_agent.models.message import AgentMessage, GeneratedScript, MessageType
from script_agent.config.settings import settings


class ScriptGenerationSkill(BaseSkill):
    """话术生成 Skill — 核心业务"""

    name = "script_generation"
    display_name = "话术生成"
    description = "根据品类、场景、达人风格生成话术"
    required_slots = ["category", "scenario"]
    input_schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "minLength": 1, "maxLength": 32},
            "scenario": {"type": "string", "minLength": 1, "maxLength": 64},
            "sub_scenario": {"type": "string", "maxLength": 64},
            "event": {"type": "string", "maxLength": 64},
            "requirements": {"type": "string", "maxLength": 2000},
            "style_hint": {"type": "string", "maxLength": 200},
            "product_name": {"type": "string", "maxLength": 128},
            "product_id": {"type": "string", "maxLength": 64},
            "target_name": {"type": "string", "maxLength": 64},
            "brand": {"type": "string", "maxLength": 64},
            "price_range": {"type": "string", "maxLength": 64},
            "target_audience": {"type": "string", "maxLength": 128},
            "selling_points": {
                "type": ["array", "string"],
                "items": {"type": "string", "maxLength": 120},
            },
            "product_features": {
                "type": ["array", "string"],
                "items": {"type": "string", "maxLength": 120},
            },
            "_raw_query": {"type": "string", "maxLength": 2000},
            "_continuation": {"type": "boolean"},
            "_category_source": {"type": "string", "maxLength": 32},
            "intent": {"type": "string", "maxLength": 64},
            "target_id": {"type": "string", "maxLength": 64},
        },
        "required": ["category", "scenario"],
        "additionalProperties": False,
    }

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
        product = context.extra.get("product")
        memory_hits = context.extra.get("memory_hits", [])

        # 生成
        script_msg = AgentMessage(
            trace_id=context.trace_id,
            payload={
                "slots": slots,
                "profile": profile,
                "product": product,
                "memory_hits": memory_hits,
                "session": session,
            },
            session_id=session.session_id,
        )
        script_resp = await self._script_agent(script_msg)
        if script_resp.message_type == MessageType.ERROR:
            return SkillResult(
                success=False,
                message=script_resp.payload.get("error_message", "话术生成失败"),
            )
        script: GeneratedScript = script_resp.payload.get("script", GeneratedScript())
        min_chars = max(1, settings.llm.script_min_chars)
        if not script.content.strip():
            return SkillResult(
                success=False,
                script=script,
                message="生成结果为空，请检查模型服务或提示词约束",
            )
        if len(script.content.strip()) < min_chars:
            return SkillResult(
                success=False,
                script=script,
                message=f"生成结果不足{min_chars}字，请重试",
            )

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
            if quality_resp.message_type == MessageType.ERROR:
                return SkillResult(
                    success=False,
                    script=script,
                    message=quality_resp.payload.get("error_message", "质量检查失败"),
                )
            quality_result = quality_resp.payload.get("quality_result")

            if quality_result and quality_result.passed:
                break

            retry_count += 1
            if retry_count <= settings.quality.max_retries:
                script_msg.payload["feedback"] = (
                    quality_result.suggestions if quality_result else []
                )
                script_resp = await self._script_agent(script_msg)
                if script_resp.message_type == MessageType.ERROR:
                    return SkillResult(
                        success=False,
                        script=script,
                        message=script_resp.payload.get("error_message", "话术生成失败"),
                    )
                script = script_resp.payload.get("script", script)
                if len(script.content.strip()) < min_chars:
                    return SkillResult(
                        success=False,
                        script=script,
                        message=f"重试后结果仍不足{min_chars}字",
                    )

        return SkillResult(
            success=quality_result.passed if quality_result else False,
            script=script,
            quality_result=quality_result,
        )
