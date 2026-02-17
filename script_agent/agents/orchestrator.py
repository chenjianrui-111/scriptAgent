"""
基于 LangGraph 的编排器

目标:
  - 将原手工状态机编排升级为可声明式图编排
  - 保留对外接口兼容 (handle_request / handle_stream)
  - 提供会话级 checkpoint，支持中断后的快速恢复
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, TypedDict

from script_agent.agents.intent_agent import IntentRecognitionAgent
from script_agent.agents.product_agent import ProductAgent
from script_agent.agents.profile_agent import ProfileAgent
from script_agent.agents.quality_agent import QualityCheckAgent
from script_agent.agents.script_agent import ScriptGenerationAgent
from script_agent.config.settings import settings
from script_agent.models.context import (
    InfluencerProfile,
    ProductProfile,
    SessionContext,
    StyleProfile,
)
from script_agent.models.message import (
    AgentMessage,
    GeneratedScript,
    IntentResult,
    MessageType,
    QualityResult,
)
from script_agent.models.state_machine import WorkflowState
from script_agent.observability import metrics as obs
from script_agent.services.long_term_memory import LongTermMemoryRetriever
from script_agent.skills.base import SkillContext, SkillResult
from script_agent.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - 环境未安装时降级
    END = "__end__"
    START = "__start__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


CheckpointSaver = Callable[[SessionContext], Awaitable[None]]
CheckpointLoader = Callable[[str], Awaitable[Optional[Dict[str, Any]]]]
CheckpointWriter = Callable[
    [str, Dict[str, Any], str, str],
    Awaitable[Optional[Dict[str, Any]]],
]


class OrchestrationState(TypedDict, total=False):
    query: str
    trace_id: str
    session: SessionContext
    checkpoint_saver: Optional[CheckpointSaver]
    checkpoint_writer: Optional[CheckpointWriter]

    workflow_start: float
    timing: Dict[str, float]
    state_history: list[str]
    current_state: str

    intent_result: IntentResult
    profile: InfluencerProfile
    product: ProductProfile
    memory_hits: list[Dict[str, Any]]
    script: GeneratedScript
    quality_result: QualityResult
    quality_feedback: list[str]
    retry_count: int
    should_retry: bool
    needs_clarification: bool
    skill_name: str
    skill_result: SkillResult
    request_recorded: bool

    result: Dict[str, Any]
    error: str


class Orchestrator:
    """中央编排器 (LangGraph)"""

    def __init__(self):
        self.memory_retriever = LongTermMemoryRetriever()
        self.intent_agent = IntentRecognitionAgent()
        self.profile_agent = ProfileAgent()
        self.product_agent = ProductAgent(memory=self.memory_retriever)
        self.script_agent = ScriptGenerationAgent()
        self.quality_agent = QualityCheckAgent()

        self.skill_registry = SkillRegistry()
        self._register_default_skills()

        self._graph = self._build_graph() if LANGGRAPH_AVAILABLE else None
        if settings.orchestration.langgraph_required and self._graph is None:
            raise RuntimeError(
                "LANGGRAPH_REQUIRED=true, but langgraph is not installed"
            )
        if self._graph is None:
            logger.warning(
                "LangGraph not installed, orchestrator will run with sequential fallback"
            )

    def _register_default_skills(self):
        try:
            from script_agent.skills.builtin.batch_generate import BatchGenerateSkill
            from script_agent.skills.builtin.script_gen import ScriptGenerationSkill
            from script_agent.skills.builtin.script_modify import ScriptModificationSkill

            self.skill_registry.register(ScriptGenerationSkill())
            self.skill_registry.register(ScriptModificationSkill())
            self.skill_registry.register(BatchGenerateSkill())
        except Exception as exc:  # pragma: no cover - 防御分支
            logger.warning(f"Failed to register default skills: {exc}")

    def _build_graph(self):
        if not LANGGRAPH_AVAILABLE or StateGraph is None:
            return None

        builder = StateGraph(OrchestrationState)

        builder.add_node("context_loading", self._node_context_loading)
        builder.add_node("intent_recognizing", self._node_intent_recognizing)
        builder.add_node("intent_clarifying", self._node_intent_clarifying)
        builder.add_node("skill_executing", self._node_skill_executing)
        builder.add_node("profile_fetching", self._node_profile_fetching)
        builder.add_node("product_fetching", self._node_product_fetching)
        builder.add_node("script_generating", self._node_script_generating)
        builder.add_node("quality_checking", self._node_quality_checking)
        builder.add_node("completed", self._node_completed)

        builder.add_edge(START, "context_loading")
        builder.add_edge("context_loading", "intent_recognizing")

        builder.add_conditional_edges(
            "intent_recognizing",
            self._route_after_intent,
            {
                "intent_clarifying": "intent_clarifying",
                "skill_executing": "skill_executing",
                "profile_fetching": "profile_fetching",
            },
        )

        builder.add_edge("intent_clarifying", "completed")
        builder.add_edge("skill_executing", "completed")
        builder.add_edge("profile_fetching", "product_fetching")
        builder.add_edge("product_fetching", "script_generating")
        builder.add_edge("script_generating", "quality_checking")

        builder.add_conditional_edges(
            "quality_checking",
            self._route_after_quality,
            {
                "script_generating": "script_generating",
                "completed": "completed",
            },
        )

        builder.add_edge("completed", END)
        return builder.compile()

    async def handle_request(
        self,
        query: str,
        session: SessionContext,
        trace_id: Optional[str] = None,
        checkpoint_saver: Optional[CheckpointSaver] = None,
        checkpoint_loader: Optional[CheckpointLoader] = None,
        checkpoint_writer: Optional[CheckpointWriter] = None,
    ) -> Dict[str, Any]:
        """处理用户请求主流程"""
        trace_id = trace_id or str(uuid.uuid4())
        workflow_start = time.perf_counter()

        workflow_snapshot = await self._load_workflow_snapshot(
            session=session,
            checkpoint_loader=checkpoint_loader,
        )

        cached = self._build_cached_completed_result(
            snapshot=workflow_snapshot,
            query=query,
            trace_id=trace_id,
        )
        if cached is not None:
            return cached

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

        initial_state: OrchestrationState = {
            "query": query,
            "trace_id": trace_id,
            "session": session,
            "checkpoint_saver": checkpoint_saver,
            "checkpoint_writer": checkpoint_writer,
            "workflow_start": workflow_start,
            "timing": {},
            "state_history": [],
            "current_state": WorkflowState.INIT.value,
            "retry_count": 0,
            "should_retry": False,
            "request_recorded": False,
            "result": result,
        }
        initial_state.update(self._build_resume_seed(query, workflow_snapshot))

        try:
            if self._graph is not None:
                final_state = await self._graph.ainvoke(initial_state)
            else:
                final_state = await self._run_without_langgraph(initial_state)
            return final_state.get("result", result)
        except Exception as exc:
            logger.error(f"[{trace_id}] Orchestrator error: {exc}", exc_info=True)
            result["error"] = str(exc)
            intent = initial_state.get("intent_result")
            obs.record_request(
                intent.intent if intent else "unknown",
                "error",
                initial_state.get("skill_name", "default"),
            )
            result["timing"]["total"] = (time.perf_counter() - workflow_start) * 1000
            result["state_history"] = list(session.state_history)
            await self._write_checkpoint(
                {
                    **initial_state,
                    "current_state": WorkflowState.ERROR.value,
                    "error": str(exc),
                },
                status="failed",
            )
            return result

    async def _run_without_langgraph(
        self,
        state: OrchestrationState,
    ) -> OrchestrationState:
        """当 langgraph 不可用时的兼容执行路径"""
        state.update(await self._node_context_loading(state))
        state.update(await self._node_intent_recognizing(state))

        route = self._route_after_intent(state)
        if route == "intent_clarifying":
            state.update(await self._node_intent_clarifying(state))
            state.update(await self._node_completed(state))
            return state

        if route == "skill_executing":
            state.update(await self._node_skill_executing(state))
            state.update(await self._node_completed(state))
            return state

        state.update(await self._node_profile_fetching(state))
        state.update(await self._node_product_fetching(state))
        while True:
            state.update(await self._node_script_generating(state))
            state.update(await self._node_quality_checking(state))
            if self._route_after_quality(state) == "completed":
                break
        state.update(await self._node_completed(state))
        return state

    async def _node_context_loading(
        self,
        state: OrchestrationState,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        history, current_state = self._move_state(state, WorkflowState.CONTEXT_LOADING)
        timing = dict(state.get("timing", {}))
        timing["context_loading"] = (time.perf_counter() - t0) * 1000

        updates = {
            "state_history": history,
            "current_state": current_state,
            "timing": timing,
        }
        await self._write_checkpoint({**state, **updates}, status="in_progress")
        return updates

    async def _node_intent_recognizing(
        self,
        state: OrchestrationState,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        history, current_state = self._move_state(state, WorkflowState.INTENT_RECOGNIZING)

        session = state["session"]
        query = state["query"]
        trace_id = state["trace_id"]

        intent_result = state.get("intent_result")
        if intent_result is None:
            intent_msg = AgentMessage(
                trace_id=trace_id,
                payload={"query": query, "session": session},
                session_id=session.session_id,
                tenant_id=session.tenant_id,
            )
            intent_response = await self.intent_agent(intent_msg)
            self._raise_if_agent_error(intent_response, "intent_recognition")
            intent_result = intent_response.payload.get(
                "intent_result", IntentResult(intent="unknown", confidence=0.0)
            )

        timing = dict(state.get("timing", {}))
        timing["intent_recognition"] = (time.perf_counter() - t0) * 1000

        obs.record_intent_confidence(intent_result.intent, intent_result.confidence)
        logger.info(
            f"[{trace_id}] Intent: {intent_result.intent} confidence={intent_result.confidence:.2f}"
        )

        skill_name = ""
        if not intent_result.needs_clarification:
            skill = self.skill_registry.route(intent_result.intent, intent_result.slots)
            if skill:
                skill_name = skill.name

        updates = {
            "state_history": history,
            "current_state": current_state,
            "timing": timing,
            "intent_result": intent_result,
            "skill_name": skill_name,
            "needs_clarification": bool(intent_result.needs_clarification),
        }
        await self._write_checkpoint({**state, **updates}, status="in_progress")
        return updates

    def _route_after_intent(self, state: OrchestrationState) -> str:
        intent_result = state.get("intent_result")
        if not intent_result:
            return "intent_clarifying"
        if intent_result.needs_clarification:
            return "intent_clarifying"
        if state.get("skill_name"):
            return "skill_executing"
        return "profile_fetching"

    async def _node_intent_clarifying(
        self,
        state: OrchestrationState,
    ) -> Dict[str, Any]:
        history, current_state = self._move_state(state, WorkflowState.INTENT_CLARIFYING)
        intent_result = state.get("intent_result", IntentResult(intent="unknown", confidence=0.0))

        result = dict(state.get("result", {}))
        result["clarification_needed"] = True
        result["clarification_question"] = (
            intent_result.clarification_question or "请提供更多信息以便生成话术"
        )

        updates = {
            "state_history": history,
            "current_state": current_state,
            "result": result,
        }
        await self._write_checkpoint({**state, **updates}, status="in_progress")
        return updates

    async def _node_skill_executing(
        self,
        state: OrchestrationState,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        history, current_state = self._move_state(state, WorkflowState.SCRIPT_GENERATING)

        session = state["session"]
        trace_id = state["trace_id"]
        query = state["query"]
        intent_result = state.get("intent_result", IntentResult(intent="unknown", confidence=0.0))

        skill_name = state.get("skill_name", "")
        skill = self.skill_registry.get(skill_name)
        if skill is None:
            raise RuntimeError(f"skill not found: {skill_name}")

        actor_role = self._resolve_actor_role(session, intent_result)
        preflight_error = self.skill_registry.preflight(
            skill=skill,
            slots=intent_result.slots,
            tenant_id=session.tenant_id,
            role=actor_role,
            query=query,
        )
        if preflight_error:
            logger.warning(
                "[%s] skill preflight blocked skill=%s reason=%s",
                trace_id,
                skill_name,
                preflight_error,
            )
            timing = dict(state.get("timing", {}))
            timing["skill_execution"] = (time.perf_counter() - t0) * 1000
            denied_result = SkillResult(success=False, message=preflight_error)
            result = dict(state.get("result", {}))
            result["success"] = False
            result["error"] = preflight_error
            result["skill_used"] = skill_name
            result["intent"] = intent_result

            updates = {
                "state_history": history,
                "current_state": current_state,
                "timing": timing,
                "skill_result": denied_result,
                "result": result,
            }
            await self._write_checkpoint({**state, **updates}, status="failed")
            return updates

        profile = state.get("profile")
        if profile is None:
            profile = await self.profile_agent.fetch(intent_result.slots)

        product = state.get("product")
        memory_hits = list(state.get("memory_hits", []))
        if intent_result.intent in {"script_generation", "script_optimization"}:
            product, memory_hits = await self.product_agent.fetch(
                intent_result.slots,
                profile,
                session,
                query=query,
            )
            if product.name:
                session.entity_cache.update(
                    "product",
                    product.product_id or product.name,
                    product.name,
                )
                intent_result.slots.setdefault("product_name", product.name)
                session.slot_context.update(intent_result.intent, intent_result.slots)

        skill_ctx = SkillContext(
            intent=intent_result,
            profile=profile,
            session=session,
            trace_id=trace_id,
            query=query,
            role=actor_role,
            extra={
                "product": product,
                "memory_hits": memory_hits,
            },
        )
        skill_result = await skill.execute(skill_ctx)

        timing = dict(state.get("timing", {}))
        timing["skill_execution"] = (time.perf_counter() - t0) * 1000

        result = dict(state.get("result", {}))
        result["success"] = bool(skill_result.success)
        result["script"] = skill_result.script
        result["quality_result"] = skill_result.quality_result
        result["intent"] = intent_result
        result["skill_used"] = skill_name
        if not skill_result.success and skill_result.message:
            result["error"] = skill_result.message

        if skill_result.script:
            session.add_turn(
                user_message=query,
                assistant_message=skill_result.script.content,
                intent=intent_result,
                generated_script=skill_result.script.content,
            )
            session.generated_scripts.append(skill_result.script)

        obs.record_skill_hit(skill_name)

        updates = {
            "state_history": history,
            "current_state": current_state,
            "timing": timing,
            "profile": profile,
            "product": product,
            "memory_hits": memory_hits,
            "script": skill_result.script,
            "quality_result": skill_result.quality_result,
            "skill_result": skill_result,
            "result": result,
        }
        await self._write_checkpoint({**state, **updates}, status="in_progress")
        return updates

    def _resolve_actor_role(
        self,
        session: SessionContext,
        intent_result: IntentResult,
    ) -> str:
        slots = intent_result.slots if intent_result else {}
        role = ""
        if isinstance(slots, dict):
            role = str(slots.get("_role", "")).strip()
        if not role:
            role = str((session.workflow_snapshot or {}).get("actor_role", "")).strip()
        if not role:
            role = settings.tool_security.default_role or "user"
        return role

    async def _node_profile_fetching(
        self,
        state: OrchestrationState,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        history, current_state = self._move_state(state, WorkflowState.PROFILE_FETCHING)

        session = state["session"]
        intent_result = state.get("intent_result", IntentResult(intent="unknown", confidence=0.0))
        slots = intent_result.slots

        profile = state.get("profile")
        if profile is None:
            profile_msg = AgentMessage(
                trace_id=state["trace_id"],
                payload={"slots": slots},
                session_id=session.session_id,
            )
            profile_response = await self.profile_agent(profile_msg)
            self._raise_if_agent_error(profile_response, "profile")
            profile = profile_response.payload.get("profile", InfluencerProfile())

        timing = dict(state.get("timing", {}))
        timing["profile_fetching"] = (time.perf_counter() - t0) * 1000

        if slots.get("target_name"):
            session.entity_cache.update(
                "influencer",
                profile.influencer_id,
                slots["target_name"],
            )
        session.slot_context.update(intent_result.intent, slots)

        updates = {
            "state_history": history,
            "current_state": current_state,
            "timing": timing,
            "profile": profile,
        }
        await self._write_checkpoint({**state, **updates}, status="in_progress")
        return updates

    async def _node_product_fetching(
        self,
        state: OrchestrationState,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        history, current_state = self._move_state(state, WorkflowState.PRODUCT_FETCHING)

        session = state["session"]
        intent_result = state.get("intent_result", IntentResult(intent="unknown", confidence=0.0))
        profile = state.get("profile", InfluencerProfile())

        product = state.get("product")
        memory_hits = list(state.get("memory_hits", []))
        if product is None:
            product, memory_hits = await self.product_agent.fetch(
                intent_result.slots,
                profile,
                session,
                query=state.get("query", ""),
            )

        if product and product.name:
            session.entity_cache.update(
                "product",
                product.product_id or product.name,
                product.name,
            )
            intent_result.slots.setdefault("product_name", product.name)
            session.slot_context.update(intent_result.intent, intent_result.slots)

        timing = dict(state.get("timing", {}))
        timing["product_fetching"] = (time.perf_counter() - t0) * 1000

        updates = {
            "state_history": history,
            "current_state": current_state,
            "timing": timing,
            "product": product,
            "memory_hits": memory_hits,
            "intent_result": intent_result,
        }
        await self._write_checkpoint({**state, **updates}, status="in_progress")
        return updates

    async def _node_script_generating(
        self,
        state: OrchestrationState,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        history, current_state = self._move_state(state, WorkflowState.SCRIPT_GENERATING)

        session = state["session"]
        intent_result = state.get("intent_result", IntentResult(intent="unknown", confidence=0.0))
        profile = state.get("profile", InfluencerProfile())
        product = state.get("product", ProductProfile())
        memory_hits = list(state.get("memory_hits", []))

        should_reuse_script = bool(state.get("script")) and bool(
            state.get("quality_result") is None and state.get("retry_count", 0) == 0
        )
        if should_reuse_script:
            script = state["script"]
        else:
            payload = {
                "slots": intent_result.slots,
                "profile": profile,
                "product": product,
                "memory_hits": memory_hits,
                "session": session,
            }
            if state.get("quality_feedback"):
                payload["feedback"] = state["quality_feedback"]
            script_msg = AgentMessage(
                trace_id=state["trace_id"],
                payload=payload,
                session_id=session.session_id,
            )
            script_response = await self.script_agent(script_msg)
            self._raise_if_agent_error(script_response, "script_generation")
            script = script_response.payload.get("script", GeneratedScript())

        timing = dict(state.get("timing", {}))
        timing["script_generation"] = (time.perf_counter() - t0) * 1000

        updates = {
            "state_history": history,
            "current_state": current_state,
            "timing": timing,
            "product": product,
            "memory_hits": memory_hits,
            "script": script,
            "quality_result": None,
        }
        await self._write_checkpoint({**state, **updates}, status="in_progress")
        return updates

    async def _node_quality_checking(
        self,
        state: OrchestrationState,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        history, current_state = self._move_state(state, WorkflowState.QUALITY_CHECKING)

        session = state["session"]
        script = state.get("script", GeneratedScript())
        profile = state.get("profile", InfluencerProfile())

        quality_msg = AgentMessage(
            trace_id=state["trace_id"],
            payload={"script": script, "profile": profile},
            session_id=session.session_id,
        )
        quality_response = await self.quality_agent(quality_msg)
        self._raise_if_agent_error(quality_response, "quality_check")
        quality_result = quality_response.payload.get("quality_result", QualityResult())

        retry_count = int(state.get("retry_count", 0))
        should_retry = False
        quality_feedback: list[str] = []

        if quality_result.passed:
            should_retry = False
        else:
            retry_count += 1
            quality_feedback = list(quality_result.suggestions)
            should_retry = retry_count <= settings.quality.max_retries
            if should_retry:
                logger.info(f"[{state['trace_id']}] Quality check failed, retry {retry_count}")

        timing = dict(state.get("timing", {}))
        timing["quality_check"] = (time.perf_counter() - t0) * 1000

        updates = {
            "state_history": history,
            "current_state": current_state,
            "timing": timing,
            "quality_result": quality_result,
            "retry_count": retry_count,
            "should_retry": should_retry,
            "quality_feedback": quality_feedback,
        }
        await self._write_checkpoint({**state, **updates}, status="in_progress")
        return updates

    def _route_after_quality(self, state: OrchestrationState) -> str:
        quality_result = state.get("quality_result")
        if quality_result and quality_result.passed:
            return "completed"
        if state.get("should_retry"):
            return "script_generating"
        return "completed"

    async def _node_completed(
        self,
        state: OrchestrationState,
    ) -> Dict[str, Any]:
        session = state["session"]
        query = state["query"]
        timing = dict(state.get("timing", {}))

        result = dict(state.get("result", {}))
        intent_result = state.get("intent_result")
        profile = state.get("profile")
        script = state.get("script") or result.get("script")
        quality_result = state.get("quality_result") or result.get("quality_result")
        product = state.get("product")

        final_state = WorkflowState.COMPLETED
        if state.get("needs_clarification"):
            final_state = WorkflowState.INTENT_CLARIFYING
        elif quality_result and not quality_result.passed and not state.get("should_retry"):
            final_state = WorkflowState.DEGRADED

        history, current_state = self._move_state(state, final_state)

        if script and not state.get("skill_result") and not state.get("needs_clarification"):
            session.add_turn(
                user_message=query,
                assistant_message=script.content,
                intent=intent_result,
                generated_script=script.content,
            )
            session.generated_scripts.append(script)

        success = bool(result.get("success", False))
        if state.get("skill_result") is not None:
            success = bool(result.get("success", False))
        elif quality_result is not None:
            success = bool(quality_result.passed)
        elif script:
            success = True

        if script:
            result["script"] = script
            result["quality_result"] = quality_result
        if product:
            result["product"] = product
        result["success"] = success
        if intent_result:
            result["intent"] = intent_result
        if profile:
            result["profile_name"] = profile.name

        timing["total"] = (time.perf_counter() - state["workflow_start"]) * 1000
        result["timing"] = timing
        result["state_history"] = list(history)

        if not state.get("request_recorded"):
            obs.record_request(
                intent_result.intent if intent_result else "unknown",
                "success" if result.get("success") else "error",
                result.get("skill_used", "default"),
            )
        if result.get("success") and script is not None:
            await self._remember_long_term(state, script)

        updates = {
            "state_history": history,
            "current_state": current_state,
            "timing": timing,
            "result": result,
            "request_recorded": True,
        }
        status = "completed" if result.get("success") else "failed"
        await self._write_checkpoint({**state, **updates}, status=status)
        return updates

    def _move_state(
        self,
        state: OrchestrationState,
        new_state: WorkflowState,
    ) -> tuple[list[str], str]:
        history = list(state.get("state_history", []))
        if not history or history[-1] != new_state.value:
            history.append(new_state.value)

        session = state["session"]
        session.current_state = new_state.value
        session.state_history = history

        return history, new_state.value

    async def _load_workflow_snapshot(
        self,
        session: SessionContext,
        checkpoint_loader: Optional[CheckpointLoader],
    ) -> Dict[str, Any]:
        if checkpoint_loader is not None:
            try:
                loaded = await checkpoint_loader(session.session_id)
                if isinstance(loaded, dict):
                    return loaded
            except Exception as exc:
                logger.warning("Failed to load checkpoint for %s: %s", session.session_id, exc)
        return session.workflow_snapshot or {}

    def _build_cached_completed_result(
        self,
        snapshot: Dict[str, Any],
        query: str,
        trace_id: str,
    ) -> Optional[Dict[str, Any]]:
        """同 query 的完成态请求快速返回，避免重复调用下游模型"""
        if not settings.orchestration.request_dedup_enabled:
            return None

        if snapshot.get("status") != "completed":
            return None
        if snapshot.get("last_query") != query:
            return None
        if not self._is_snapshot_fresh(snapshot):
            return None

        script = self._to_script(snapshot.get("script"))
        if script is None:
            return None
        quality = self._to_quality(snapshot.get("quality_result"))
        intent = self._to_intent_result(snapshot.get("intent"))
        product = self._to_product(snapshot.get("product"))

        result: Dict[str, Any] = {
            "trace_id": trace_id,
            "success": True if quality is None else bool(quality.passed),
            "script": script,
            "quality_result": quality,
            "clarification_needed": False,
            "clarification_question": "",
            "state_history": list(snapshot.get("state_history", [])),
            "timing": {"total": 0.0, "cache_hit": 0.0},
            "from_cache": True,
        }
        if intent:
            result["intent"] = intent
        if product:
            result["product"] = product
        obs.record_workflow_cache_hit()
        obs.record_request(
            intent.intent if intent else "unknown",
            "success" if result["success"] else "error",
            "dedup_cache",
        )
        return result

    def _is_snapshot_fresh(self, snapshot: Dict[str, Any]) -> bool:
        ts = snapshot.get("updated_at")
        if not isinstance(ts, str):
            return False
        try:
            updated_at = datetime.fromisoformat(ts)
        except ValueError:
            return False
        age = (datetime.now() - updated_at).total_seconds()
        return age <= settings.orchestration.request_dedup_ttl_seconds

    def _to_intent_result(self, data: Any) -> Optional[IntentResult]:
        if not isinstance(data, dict):
            return None
        return IntentResult(
            intent=data.get("intent", "unknown"),
            confidence=float(data.get("confidence", 0.0)),
            slots=dict(data.get("slots", {})),
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_question=data.get("clarification_question", ""),
        )

    def _to_profile(self, data: Any) -> Optional[InfluencerProfile]:
        if not isinstance(data, dict):
            return None
        style_data = data.get("style", {})
        return InfluencerProfile(
            influencer_id=data.get("influencer_id", ""),
            name=data.get("name", ""),
            category=data.get("category", ""),
            audience_age_range=data.get("audience_age_range", ""),
            audience_gender_ratio=data.get("audience_gender_ratio", ""),
            top_content_keywords=list(data.get("top_content_keywords", [])),
            style=StyleProfile(
                tone=style_data.get("tone", ""),
                formality_level=float(style_data.get("formality_level", 0.5)),
                catchphrases=list(style_data.get("catchphrases", [])),
                avg_sentence_length=float(style_data.get("avg_sentence_length", 20.0)),
                interaction_frequency=float(style_data.get("interaction_frequency", 0.5)),
                humor_level=float(style_data.get("humor_level", 0.3)),
            ),
        )

    def _to_product(self, data: Any) -> Optional[ProductProfile]:
        if not isinstance(data, dict):
            return None
        return ProductProfile(
            product_id=data.get("product_id", ""),
            name=data.get("name", ""),
            category=data.get("category", ""),
            brand=data.get("brand", ""),
            price_range=data.get("price_range", ""),
            features=list(data.get("features", [])),
            selling_points=list(data.get("selling_points", [])),
            target_audience=data.get("target_audience", ""),
            compliance_notes=list(data.get("compliance_notes", [])),
        )

    def _to_script(self, data: Any) -> Optional[GeneratedScript]:
        if not isinstance(data, dict):
            return None
        if data.get("content_truncated"):
            # 截断快照不参与恢复，避免半内容污染结果
            return None
        return GeneratedScript(
            script_id=data.get("script_id") or str(uuid.uuid4()),
            content=data.get("content", ""),
            category=data.get("category", ""),
            scenario=data.get("scenario", ""),
            style_keywords=list(data.get("style_keywords", [])),
            turn_index=int(data.get("turn_index", 0)),
            adopted=bool(data.get("adopted", False)),
            quality_score=float(data.get("quality_score", 0.0)),
            generation_params=dict(data.get("generation_params", {})),
        )

    def _to_quality(self, data: Any) -> Optional[QualityResult]:
        if not isinstance(data, dict):
            return None
        return QualityResult(
            passed=bool(data.get("passed", False)),
            overall_score=float(data.get("overall_score", 0.0)),
            sensitive_words=list(data.get("sensitive_words", [])),
            compliance_issues=list(data.get("compliance_issues", [])),
            style_consistency=float(data.get("style_consistency", 0.0)),
            suggestions=list(data.get("suggestions", [])),
        )

    def _raise_if_agent_error(self, response: AgentMessage, agent_name: str) -> None:
        if response.message_type != MessageType.ERROR:
            return
        error_msg = response.payload.get("error_message", "unknown error")
        raise RuntimeError(f"{agent_name} failed: {error_msg}")

    def _build_resume_seed(self, query: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """从持久化快照恢复可复用状态（同 query）"""
        if snapshot.get("last_query") != query:
            return {}
        if snapshot.get("status") not in {"in_progress", "failed"}:
            return {}
        # 兼容旧快照：仅在存在 updated_at 且判定过期时，才拒绝恢复。
        # 旧版本快照可能没有 updated_at，但仍然包含可恢复状态。
        if (
            snapshot.get("status") == "in_progress"
            and snapshot.get("updated_at")
            and not self._is_snapshot_fresh(snapshot)
        ):
            return {}

        seed: Dict[str, Any] = {}

        intent = self._to_intent_result(snapshot.get("intent"))
        if intent:
            seed["intent_result"] = intent

        profile = self._to_profile(snapshot.get("profile"))
        if profile:
            seed["profile"] = profile

        product = self._to_product(snapshot.get("product"))
        if product:
            seed["product"] = product

        script = self._to_script(snapshot.get("script"))
        if script:
            seed["script"] = script

        quality = self._to_quality(snapshot.get("quality_result"))
        if quality:
            seed["quality_result"] = quality

        if "retry_count" in snapshot:
            seed["retry_count"] = int(snapshot.get("retry_count", 0))

        logger.info(
            "Recovered workflow snapshot state=%s",
            snapshot.get("current_state", ""),
        )
        return seed

    async def _write_checkpoint(
        self,
        state: OrchestrationState,
        status: str,
    ) -> None:
        """将编排快照写入 session，必要时自动持久化"""
        session = state["session"]
        intent = state.get("intent_result")
        profile = state.get("profile")
        product = state.get("product")
        script = state.get("script")
        quality_result = state.get("quality_result")
        memory_hits = list(state.get("memory_hits", []))

        snapshot: Dict[str, Any] = {
            "version": 1,
            "status": status,
            "trace_id": state.get("trace_id", ""),
            "last_query": state.get("query", ""),
            "current_state": state.get("current_state", session.current_state),
            "state_history": list(state.get("state_history", [])),
            "retry_count": int(state.get("retry_count", 0)),
            "updated_at": datetime.now().isoformat(),
        }

        if state.get("error"):
            snapshot["error"] = state["error"]
        if intent:
            snapshot["intent"] = {
                "intent": intent.intent,
                "confidence": intent.confidence,
                "slots": dict(intent.slots),
                "needs_clarification": intent.needs_clarification,
                "clarification_question": intent.clarification_question,
            }
        if profile:
            snapshot["profile"] = {
                "influencer_id": profile.influencer_id,
                "name": profile.name,
                "category": profile.category,
                "audience_age_range": profile.audience_age_range,
                "audience_gender_ratio": profile.audience_gender_ratio,
                "top_content_keywords": list(profile.top_content_keywords),
                "style": {
                    "tone": profile.style.tone,
                    "formality_level": profile.style.formality_level,
                    "catchphrases": list(profile.style.catchphrases),
                    "avg_sentence_length": profile.style.avg_sentence_length,
                    "interaction_frequency": profile.style.interaction_frequency,
                    "humor_level": profile.style.humor_level,
                },
            }
        if product:
            snapshot["product"] = {
                "product_id": product.product_id,
                "name": product.name,
                "category": product.category,
                "brand": product.brand,
                "price_range": product.price_range,
                "features": list(product.features),
                "selling_points": list(product.selling_points),
                "target_audience": product.target_audience,
                "compliance_notes": list(product.compliance_notes),
            }
        if script:
            max_chars = settings.orchestration.checkpoint_script_max_chars
            content = script.content
            content_truncated = False
            if len(content) > max_chars:
                content = content[:max_chars]
                content_truncated = True
            snapshot["script"] = {
                "script_id": script.script_id,
                "content": content,
                "content_length": len(script.content),
                "content_truncated": content_truncated,
                "category": script.category,
                "scenario": script.scenario,
                "style_keywords": list(script.style_keywords),
                "turn_index": script.turn_index,
                "adopted": script.adopted,
                "quality_score": script.quality_score,
                "generation_params": dict(script.generation_params),
            }
        if quality_result:
            snapshot["quality_result"] = {
                "passed": quality_result.passed,
                "overall_score": quality_result.overall_score,
                "sensitive_words": list(quality_result.sensitive_words),
                "compliance_issues": list(quality_result.compliance_issues),
                "style_consistency": quality_result.style_consistency,
                "suggestions": list(quality_result.suggestions),
            }
        if memory_hits:
            snapshot["memory_hits"] = [
                {
                    "memory_id": m.get("memory_id", ""),
                    "score": round(float(m.get("score", 0.0)), 4),
                }
                for m in memory_hits[:5]
            ]

        checkpoint_writer = state.get("checkpoint_writer")
        if checkpoint_writer is not None:
            record = None
            try:
                record = await checkpoint_writer(
                    session.session_id,
                    snapshot,
                    str(state.get("trace_id", "")),
                    status,
                )
            except Exception as exc:
                logger.warning("checkpoint_writer failed for %s: %s", session.session_id, exc)

            summary = {
                "status": snapshot.get("status"),
                "trace_id": snapshot.get("trace_id"),
                "last_query": snapshot.get("last_query"),
                "current_state": snapshot.get("current_state"),
                "state_history": snapshot.get("state_history", [])[-6:],
                "retry_count": snapshot.get("retry_count", 0),
                "updated_at": snapshot.get("updated_at"),
            }
            if record and isinstance(record, dict):
                summary["version"] = record.get("version")
                summary["created_at"] = record.get("created_at")
                summary["checksum"] = record.get("checksum")
            session.mark_workflow_snapshot(summary)
        else:
            session.mark_workflow_snapshot(snapshot)

        obs.record_checkpoint(status)

        checkpoint_saver = state.get("checkpoint_saver")
        if (
            checkpoint_saver is not None
            and settings.orchestration.checkpoint_auto_save
        ):
            await checkpoint_saver(session)

    async def _remember_long_term(
        self,
        state: OrchestrationState,
        script: GeneratedScript,
    ) -> None:
        intent = state.get("intent_result")
        profile = state.get("profile")
        session = state["session"]
        product = state.get("product")
        try:
            await self.memory_retriever.remember_script(
                session=session,
                intent=intent,
                profile=profile,
                product=product,
                script=script,
                query=state.get("query", ""),
            )
        except Exception as exc:
            logger.warning("long-term memory write failed: %s", exc)

    async def handle_stream(
        self,
        query: str,
        session: SessionContext,
        trace_id: Optional[str] = None,
        checkpoint_saver: Optional[CheckpointSaver] = None,
        checkpoint_loader: Optional[CheckpointLoader] = None,
        checkpoint_writer: Optional[CheckpointWriter] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式处理：与同步链路复用同一状态推进和 checkpoint writer。
        """
        trace_id = trace_id or str(uuid.uuid4())
        workflow_start = time.perf_counter()
        workflow_snapshot = await self._load_workflow_snapshot(
            session=session,
            checkpoint_loader=checkpoint_loader,
        )

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
        state: OrchestrationState = {
            "query": query,
            "trace_id": trace_id,
            "session": session,
            "checkpoint_saver": checkpoint_saver,
            "checkpoint_writer": checkpoint_writer,
            "workflow_start": workflow_start,
            "timing": {},
            "state_history": [],
            "current_state": WorkflowState.INIT.value,
            "retry_count": 0,
            "should_retry": False,
            "request_recorded": False,
            "result": result,
        }
        state.update(self._build_resume_seed(query, workflow_snapshot))

        try:
            state.update(await self._node_context_loading(state))
            state.update(await self._node_intent_recognizing(state))

            route = self._route_after_intent(state)
            if route == "intent_clarifying":
                state.update(await self._node_intent_clarifying(state))
                state.update(await self._node_completed(state))
                question = state.get("result", {}).get(
                    "clarification_question", "请提供更多信息以便生成话术"
                )
                if question:
                    yield question
                return

            if route == "skill_executing":
                state.update(await self._node_skill_executing(state))
                state.update(await self._node_completed(state))
                result_payload = state.get("result", {})
                script = result_payload.get("script") or state.get("script")
                if script and script.content and script.content.strip():
                    yield script.content
                    return
                error_msg = str(
                    result_payload.get("error")
                    or "生成失败，请稍后重试"
                ).strip()
                yield f"[ERROR] {error_msg}"
                return

            state.update(await self._node_profile_fetching(state))
            state.update(await self._node_product_fetching(state))

            stream_start = time.perf_counter()
            history, current_state = self._move_state(state, WorkflowState.SCRIPT_GENERATING)
            timing = dict(state.get("timing", {}))
            updates = {
                "state_history": history,
                "current_state": current_state,
                "timing": timing,
            }
            state.update(updates)
            await self._write_checkpoint(state, status="in_progress")

            intent_result = state.get(
                "intent_result", IntentResult(intent="script_generation", confidence=0.5)
            )
            profile = state.get("profile", InfluencerProfile())
            product = state.get("product", ProductProfile())
            memory_hits = list(state.get("memory_hits", []))
            base_slots = dict(intent_result.slots)
            quality_feedback: List[str] = []
            min_chars = max(1, settings.llm.script_min_chars)
            max_attempts = max(1, settings.quality.max_retries + 1)
            final_chunks: List[str] = []
            final_script: Optional[GeneratedScript] = None
            final_quality: Optional[QualityResult] = None
            best_script: Optional[GeneratedScript] = None
            best_quality: Optional[QualityResult] = None

            for attempt in range(1, max_attempts + 1):
                attempt_slots = dict(base_slots)
                if quality_feedback:
                    attempt_slots["requirements"] = self._merge_requirements_feedback(
                        str(base_slots.get("requirements", "")),
                        quality_feedback,
                    )

                chunks: List[str] = []
                try:
                    async for token in self.script_agent.generate_stream(
                        attempt_slots,
                        profile,
                        session,
                        product=product,
                        memory_hits=memory_hits,
                    ):
                        chunks.append(token)
                except Exception as gen_exc:
                    logger.error(
                        "[%s] Script generate_stream failed (attempt=%s/%s): %s",
                        trace_id,
                        attempt,
                        max_attempts,
                        gen_exc,
                        exc_info=True,
                    )
                    state["error"] = str(gen_exc)
                    if attempt >= max_attempts:
                        break
                    quality_feedback = ["请只输出完整的口播正文，不要输出提示词或结构化字段。"]
                    continue

                content = "".join(chunks).strip()
                if not content:
                    if attempt >= max_attempts:
                        break
                    quality_feedback = [
                        "请输出完整正文，不要留空。",
                        "请至少输出一段完整可口播文案。",
                    ]
                    continue

                script = GeneratedScript(
                    content=content,
                    category=(
                        attempt_slots.get("category")
                        or profile.category
                        or session.category
                        or "通用"
                    ),
                    scenario=attempt_slots.get("scenario", "stream_generation"),
                    style_keywords=self._extract_style_keywords(
                        attempt_slots.get("style_hint")
                    ),
                    turn_index=len(session.turns),
                    generation_params={
                        "streaming": True,
                        "memory_hits": len(memory_hits),
                        "attempt": attempt,
                        "product_name": str(attempt_slots.get("product_name", "")).strip(),
                        "product_switch": bool(attempt_slots.get("_product_switch")),
                        "previous_product_name": str(
                            attempt_slots.get("_previous_product_name", "")
                        ).strip(),
                    },
                )

                quality_msg = AgentMessage(
                    trace_id=state["trace_id"],
                    payload={"script": script, "profile": profile},
                    session_id=session.session_id,
                )
                quality_response = await self.quality_agent(quality_msg)
                self._raise_if_agent_error(quality_response, "quality_check")
                quality_result = quality_response.payload.get("quality_result", QualityResult())

                if (
                    best_script is None
                    or quality_result.overall_score > (best_quality.overall_score if best_quality else 0.0)
                ):
                    best_script = script
                    best_quality = quality_result

                if quality_result.passed and len(content) >= min_chars:
                    final_chunks = chunks
                    final_script = script
                    final_quality = quality_result
                    quality_feedback = []
                    break

                quality_feedback = list(quality_result.suggestions[:4])
                quality_feedback.append("请确保输出是完整文案，句尾完整收束，不要出现半句。")
                state["retry_count"] = max(0, attempt - 1)
                if attempt >= max_attempts:
                    break

            timing["script_generation"] = (time.perf_counter() - stream_start) * 1000
            if final_script:
                for token in final_chunks:
                    yield token
                state["script"] = final_script
                state["quality_result"] = final_quality
            elif best_script and len(best_script.content.strip()) >= min_chars:
                for token in [best_script.content]:
                    yield token
                yield "\n\n[提示] 已返回降级版本，建议点击“重新生成”获取更高质量结果。"
                state["script"] = best_script
                state["quality_result"] = best_quality or QualityResult(
                    passed=False,
                    overall_score=0.0,
                    suggestions=["建议重新生成以获取完整文案"],
                )
                state["error"] = state.get("error") or "stream quality check not passed"
            elif best_script:
                state["error"] = state.get("error") or (
                    f"stream output too short: {len(best_script.content.strip())} < {min_chars}"
                )
                yield "\n\n[提示] 生成内容过短，建议重新生成或补充更多商品信息"
            elif not state.get("error"):
                state["error"] = "empty stream output"
                yield "[生成失败] 未能生成有效话术内容，请检查商品信息后重试"

            state["timing"] = timing
            state.update(await self._node_completed(state))
        except Exception as exc:
            logger.error(f"[{trace_id}] Stream orchestrator error: {exc}", exc_info=True)
            state["error"] = str(exc)
            state["current_state"] = WorkflowState.ERROR.value
            result = dict(state.get("result", {}))
            result["error"] = str(exc)
            result["success"] = False
            result["timing"] = {
                **dict(state.get("timing", {})),
                "total": (time.perf_counter() - workflow_start) * 1000,
            }
            result["state_history"] = list(session.state_history)
            state["result"] = result
            await self._write_checkpoint(state, status="failed")
            raise

    def _extract_style_keywords(self, style_hint: Any) -> list[str]:
        if isinstance(style_hint, str):
            normalized = style_hint.replace("，", ",")
            return [x.strip() for x in normalized.split(",") if x.strip()]
        if isinstance(style_hint, list):
            return [str(x).strip() for x in style_hint if str(x).strip()]
        return []

    def _merge_requirements_feedback(
        self,
        requirements: str,
        feedback: List[str],
    ) -> str:
        base = (requirements or "").strip()
        tips = [str(x).strip() for x in feedback if str(x).strip()]
        if not tips:
            return base
        merged = "；".join(dict.fromkeys(tips))
        if base:
            return f"{base}；质量修正要求：{merged}"[:2000]
        return f"质量修正要求：{merged}"[:2000]

    def info(self) -> Dict[str, Any]:
        return {
            "langgraph_enabled": self._graph is not None,
            "dedup_enabled": settings.orchestration.request_dedup_enabled,
            "checkpoint_auto_save": settings.orchestration.checkpoint_auto_save,
            "longterm_memory_enabled": settings.longterm_memory.enabled,
            "longterm_memory_backend": settings.longterm_memory.backend,
        }

    async def shutdown(self) -> None:
        """释放底层连接资源"""
        clients = []
        for agent in (
            self.intent_agent,
            self.profile_agent,
            self.product_agent,
            self.script_agent,
            self.quality_agent,
        ):
            llm = getattr(agent, "llm", None)
            if llm is not None:
                clients.append(llm)

        # 关闭 skill 内部可能持有的 LLM client
        for skill in self.skill_registry.iter_skills():
            for attr in ("_llm", "_script_agent", "_quality_agent"):
                obj = getattr(skill, attr, None)
                if obj is None:
                    continue
                if hasattr(obj, "close"):
                    clients.append(obj)
                llm = getattr(obj, "llm", None)
                if llm is not None:
                    clients.append(llm)

        unique_clients = list({id(c): c for c in clients}.values())
        for client in unique_clients:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    logger.warning("Failed to close client: %s", type(client).__name__)
        try:
            await self.memory_retriever.close()
        except Exception:
            logger.warning("Failed to close LongTermMemoryRetriever")
