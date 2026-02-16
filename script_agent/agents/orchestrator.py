"""
编排器 (Orchestrator) - 中央工作流协调器

编排器模式优势: 可观测性强、流程灵活、解耦彻底
状态机驱动: INIT→INTENT→PROFILE→SCRIPT→QUALITY→COMPLETED
支持: 条件分支、重试、降级、并行执行
"""

import asyncio
import uuid
import time
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from script_agent.agents.base import BaseAgent
from script_agent.agents.intent_agent import IntentRecognitionAgent
from script_agent.agents.profile_agent import ProfileAgent
from script_agent.agents.script_agent import ScriptGenerationAgent
from script_agent.agents.quality_agent import QualityCheckAgent
from script_agent.models.message import (
    AgentMessage, MessageType, IntentResult,
    GeneratedScript, QualityResult,
)
from script_agent.models.state_machine import (
    StateMachine, StateContext, WorkflowState,
)
from script_agent.models.context import (
    SessionContext, InfluencerProfile,
)
from script_agent.config.settings import settings
from script_agent.skills.registry import SkillRegistry
from script_agent.skills.base import SkillContext
from script_agent.observability import metrics as obs

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    编排器 - 管理四个子Agent的协作流程 + Skill 路由

    流程:
      1. 意图识别 → 2. Skill 路由 (命中则走 Skill)
         ↓ (未命中)
      3. 画像获取 → 4. 话术生成 → 5. 质量校验
    """

    def __init__(self):
        # 子Agents
        self.intent_agent = IntentRecognitionAgent()
        self.profile_agent = ProfileAgent()
        self.script_agent = ScriptGenerationAgent()
        self.quality_agent = QualityCheckAgent()

        # 状态机
        self.state_machine = StateMachine()

        # Skill 系统
        self.skill_registry = SkillRegistry()
        self._register_default_skills()

    def _register_default_skills(self):
        """注册内置 Skills"""
        try:
            from script_agent.skills.builtin.script_gen import ScriptGenerationSkill
            from script_agent.skills.builtin.script_modify import ScriptModificationSkill
            from script_agent.skills.builtin.batch_generate import BatchGenerateSkill

            self.skill_registry.register(ScriptGenerationSkill())
            self.skill_registry.register(ScriptModificationSkill())
            self.skill_registry.register(BatchGenerateSkill())
        except Exception as e:
            logger.warning(f"Failed to register default skills: {e}")

    async def handle_request(self, query: str,
                              session: SessionContext,
                              trace_id: Optional[str] = None
                              ) -> Dict[str, Any]:
        """
        处理用户请求的完整流程

        Args:
            query: 用户输入
            session: 当前会话上下文
            trace_id: 链路追踪ID

        Returns:
            包含生成结果、质量报告等的字典
        """
        trace_id = trace_id or str(uuid.uuid4())
        state_ctx = StateContext()
        workflow_start = time.perf_counter()

        result: Dict[str, Any] = {
            "trace_id": trace_id,
            "success": False,
            "script": None,
            "quality_result": None,
            "clarification_needed": False,
            "clarification_question": "",
            "state_history": [],
            "timing": {},
        }

        try:
            # ============================================================
            # Step 1: INIT → CONTEXT_LOADING
            # ============================================================
            self.state_machine.transition(state_ctx, {})
            t0 = time.perf_counter()
            # 会话已在外部加载, 这里仅做标记
            result["timing"]["context_loading"] = (time.perf_counter() - t0) * 1000

            # ============================================================
            # Step 2: CONTEXT_LOADING → INTENT_RECOGNIZING
            # ============================================================
            self.state_machine.transition(state_ctx, {})
            t0 = time.perf_counter()

            intent_msg = AgentMessage(
                trace_id=trace_id,
                payload={"query": query, "session": session},
                session_id=session.session_id,
                tenant_id=session.tenant_id,
            )
            intent_response = await self.intent_agent(intent_msg)
            intent_result: IntentResult = intent_response.payload.get(
                "intent_result", IntentResult(intent="unknown", confidence=0.0)
            )

            result["timing"]["intent_recognition"] = (time.perf_counter() - t0) * 1000
            obs.record_intent_confidence(intent_result.intent, intent_result.confidence)
            logger.info(
                f"[{trace_id}] Intent: {intent_result.intent} "
                f"confidence={intent_result.confidence:.2f}"
            )

            # ============================================================
            # Step 3: 意图分支判断
            # ============================================================
            conditions = {
                "confidence": intent_result.confidence,
                "intent": intent_result.intent,
            }
            next_state = self.state_machine.transition(state_ctx, conditions)

            # 如果需要澄清 → 返回澄清问题
            if (intent_result.needs_clarification
                    or next_state == WorkflowState.INTENT_CLARIFYING):
                result["clarification_needed"] = True
                result["clarification_question"] = (
                    intent_result.clarification_question
                    or "请提供更多信息以便生成话术"
                )
                result["state_history"] = [s.value for s in state_ctx.state_history]
                return result

            # ============================================================
            # Step 3.5: Skill 路由 (命中则走 Skill 快速通道)
            # ============================================================
            slots = intent_result.slots
            skill = self.skill_registry.route(intent_result.intent, slots)

            if skill:
                t0 = time.perf_counter()
                # Skill 需要画像, 先获取
                profile = await self.profile_agent.fetch(slots)
                skill_ctx = SkillContext(
                    intent=intent_result,
                    profile=profile,
                    session=session,
                    trace_id=trace_id,
                )
                skill_result = await skill.execute(skill_ctx)
                result["timing"]["skill_execution"] = (time.perf_counter() - t0) * 1000
                result["success"] = skill_result.success
                result["script"] = skill_result.script
                result["quality_result"] = skill_result.quality_result
                result["intent"] = intent_result
                result["skill_used"] = skill.name
                obs.record_skill_hit(skill.name)
                obs.record_request(intent_result.intent, "success" if skill_result.success else "error", skill.name)
                if skill_result.script:
                    session.add_turn(
                        user_message=query,
                        assistant_message=skill_result.script.content,
                        intent=intent_result,
                        generated_script=skill_result.script.content,
                    )
                    session.generated_scripts.append(skill_result.script)
                result["timing"]["total"] = (time.perf_counter() - workflow_start) * 1000
                result["state_history"] = [s.value for s in state_ctx.state_history]
                return result

            # ============================================================
            # Step 4: PROFILE_FETCHING (默认流程, 无 Skill 匹配)
            # ============================================================
            t0 = time.perf_counter()

            profile_msg = AgentMessage(
                trace_id=trace_id,
                payload={"slots": slots},
                session_id=session.session_id,
            )
            profile_response = await self.profile_agent(profile_msg)
            profile: InfluencerProfile = profile_response.payload.get(
                "profile", InfluencerProfile()
            )

            result["timing"]["profile_fetching"] = (time.perf_counter() - t0) * 1000

            # 更新Entity缓存 (供后续轮次指代消解)
            if slots.get("target_name"):
                session.entity_cache.update(
                    "influencer",
                    profile.influencer_id,
                    slots["target_name"],
                )
            session.slot_context.update(intent_result.intent, slots)

            # ============================================================
            # Step 5: SCRIPT_GENERATING
            # ============================================================
            self.state_machine.transition(state_ctx, {"confidence": 1.0})
            t0 = time.perf_counter()

            script_msg = AgentMessage(
                trace_id=trace_id,
                payload={
                    "slots": slots,
                    "profile": profile,
                    "session": session,
                },
                session_id=session.session_id,
            )
            script_response = await self.script_agent(script_msg)
            script: GeneratedScript = script_response.payload.get(
                "script", GeneratedScript()
            )

            result["timing"]["script_generation"] = (time.perf_counter() - t0) * 1000

            # ============================================================
            # Step 6: QUALITY_CHECKING (支持重试)
            # ============================================================
            self.state_machine.transition(state_ctx, {})
            t0 = time.perf_counter()
            retry_count = 0

            while retry_count <= settings.quality.max_retries:
                quality_msg = AgentMessage(
                    trace_id=trace_id,
                    payload={"script": script, "profile": profile},
                    session_id=session.session_id,
                )
                quality_response = await self.quality_agent(quality_msg)
                quality_result: QualityResult = quality_response.payload.get(
                    "quality_result", QualityResult()
                )

                if quality_result.passed:
                    break

                retry_count += 1
                if retry_count <= settings.quality.max_retries:
                    logger.info(
                        f"[{trace_id}] Quality check failed, retry {retry_count}"
                    )
                    # 带反馈重新生成
                    script_msg.payload["feedback"] = quality_result.suggestions
                    script_response = await self.script_agent(script_msg)
                    script = script_response.payload.get("script", script)

            result["timing"]["quality_check"] = (time.perf_counter() - t0) * 1000

            # ============================================================
            # Step 7: COMPLETED
            # ============================================================
            conditions = {
                "quality_passed": quality_result.passed,
                "retry_count": retry_count,
                "max_retries": settings.quality.max_retries,
            }
            self.state_machine.transition(state_ctx, conditions)

            # 更新会话
            session.add_turn(
                user_message=query,
                assistant_message=script.content,
                intent=intent_result,
                generated_script=script.content,
            )
            session.generated_scripts.append(script)

            # 构建结果
            result["success"] = True
            result["script"] = script
            result["quality_result"] = quality_result
            result["intent"] = intent_result
            result["profile_name"] = profile.name

        except Exception as e:
            logger.error(f"[{trace_id}] Orchestrator error: {e}", exc_info=True)
            result["error"] = str(e)
            state_ctx.current_state = WorkflowState.ERROR

        # 记录总耗时
        result["timing"]["total"] = (time.perf_counter() - workflow_start) * 1000
        result["state_history"] = [s.value for s in state_ctx.state_history] + [
            state_ctx.current_state.value
        ]
        return result

    async def handle_stream(self, query: str,
                             session: SessionContext,
                             ) -> AsyncGenerator[str, None]:
        """
        流式处理 - 首token前完成所有前置步骤, 然后流式返回
        """
        # 前置步骤 (阻塞, 首token前必须完成)
        intent_msg = AgentMessage(
            payload={"query": query, "session": session},
            session_id=session.session_id,
        )
        intent_resp = await self.intent_agent(intent_msg)
        intent_result = intent_resp.payload.get("intent_result", IntentResult(
            intent="script_generation", confidence=0.5
        ))

        if intent_result.needs_clarification:
            yield intent_result.clarification_question
            return

        # 并行获取画像
        profile = await self.profile_agent.fetch(intent_result.slots)

        # 流式生成
        async for token in self.script_agent.generate_stream(
            intent_result.slots, profile, session
        ):
            yield token
