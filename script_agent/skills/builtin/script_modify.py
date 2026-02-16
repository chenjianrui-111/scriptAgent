"""
话术修改 Skill — 修改已生成的话术
"""

from typing import Any, Dict

from script_agent.skills.base import BaseSkill, SkillContext, SkillResult
from script_agent.models.message import GeneratedScript
from script_agent.services.llm_client import LLMServiceClient


class ScriptModificationSkill(BaseSkill):
    """话术修改 Skill"""

    name = "script_modification"
    display_name = "话术修改"
    description = "修改已生成的话术内容"
    required_slots = []
    input_schema = {
        "type": "object",
        "properties": {
            "requirements": {"type": "string", "maxLength": 2000},
            "_raw_query": {"type": "string", "maxLength": 2000},
            "_continuation": {"type": "boolean"},
            "_category_source": {"type": "string", "maxLength": 32},
            "category": {"type": "string", "maxLength": 32},
            "scenario": {"type": "string", "maxLength": 64},
            "sub_scenario": {"type": "string", "maxLength": 64},
            "style_hint": {"type": "string", "maxLength": 200},
            "product_name": {"type": "string", "maxLength": 128},
            "target_name": {"type": "string", "maxLength": 64},
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self):
        self._llm = LLMServiceClient()

    def can_handle(self, intent: str, slots: Dict[str, Any]) -> float:
        if intent == "script_modification":
            return 1.0
        return 0.0

    async def execute(self, context: SkillContext) -> SkillResult:
        session = context.session
        slots = context.intent.slots
        query = slots.get("requirements", "")

        # 找到要修改的话术
        if not session.generated_scripts:
            return SkillResult(
                success=False, message="没有找到可修改的话术，请先生成一段话术"
            )

        target_script = session.generated_scripts[-1]
        profile = context.profile

        prompt = f"""你是话术修改专家。请根据用户要求修改以下话术。

## 原始话术
{target_script.content}

## 达人风格
- 语气: {profile.style.tone}
- 口头禅: {', '.join(profile.style.catchphrases[:3])}

## 修改要求
{query or context.intent.slots.get('_raw_query', '优化表达')}

## 要求
- 保持达人风格一致
- 仅修改用户要求的部分
- 确保符合平台规范

请直接输出修改后的话术，不要包含其他内容。
"""
        try:
            modified = await self._llm.generate_sync(
                prompt,
                category=target_script.category or "通用",
                max_tokens=800,
            )
            new_script = GeneratedScript(
                content=modified,
                category=target_script.category,
                scenario=target_script.scenario,
                style_keywords=target_script.style_keywords,
                turn_index=len(session.turns),
            )
            return SkillResult(success=True, script=new_script)
        except Exception as e:
            return SkillResult(success=False, message=f"修改失败: {e}")
