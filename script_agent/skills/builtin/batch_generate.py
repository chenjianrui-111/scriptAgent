"""
批量生成 Skill — 一次生成多场景话术
"""

import asyncio
from typing import Any, Dict, List

from script_agent.skills.base import BaseSkill, SkillContext, SkillResult
from script_agent.models.message import AgentMessage, GeneratedScript
from script_agent.agents.script_agent import ScriptGenerationAgent


class BatchGenerateSkill(BaseSkill):
    """批量生成 Skill — 一次生成多场景话术"""

    name = "batch_generate"
    display_name = "批量生成"
    description = "一次性生成开场、产品介绍、促销等多场景话术"
    required_slots = ["category"]

    SCENARIOS = ["开场话术", "产品介绍", "促销话术", "种草文案"]

    def __init__(self):
        self._script_agent = ScriptGenerationAgent()

    def can_handle(self, intent: str, slots: Dict[str, Any]) -> float:
        if intent == "script_generation":
            query = slots.get("_raw_query", "")
            if "批量" in query or "全套" in query or "一套" in query:
                return 0.9
        return 0.0

    async def execute(self, context: SkillContext) -> SkillResult:
        slots = context.intent.slots
        profile = context.profile
        session = context.session
        product = context.extra.get("product")
        memory_hits = context.extra.get("memory_hits", [])

        async def gen_one(scenario: str) -> GeneratedScript:
            s = dict(slots)
            s["sub_scenario"] = scenario
            msg = AgentMessage(
                trace_id=context.trace_id,
                payload={
                    "slots": s,
                    "profile": profile,
                    "product": product,
                    "memory_hits": memory_hits,
                    "session": session,
                },
                session_id=session.session_id,
            )
            resp = await self._script_agent(msg)
            return resp.payload.get("script", GeneratedScript())

        scripts = await asyncio.gather(
            *[gen_one(sc) for sc in self.SCENARIOS]
        )

        combined_content = "\n\n".join(
            f"【{sc}】\n{script.content}"
            for sc, script in zip(self.SCENARIOS, scripts)
        )
        combined = GeneratedScript(
            content=combined_content,
            category=slots.get("category", ""),
            scenario="批量生成",
            turn_index=len(session.turns),
        )

        return SkillResult(
            success=True,
            script=combined,
            data={"scenario_count": len(self.SCENARIOS)},
        )
