"""
状态机 - 驱动Agent编排流程的状态流转
状态定义：INIT→INTENT→PROFILE→SCRIPT→QUALITY→COMPLETED
支持条件分支、重试、降级等复杂场景

所有条件转移使用 guard function, 类型安全, 可测试可扩展。
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class WorkflowState(str, Enum):
    """工作流状态"""
    # 主流程状态
    INIT = "INIT"
    CONTEXT_LOADING = "CONTEXT_LOADING"
    INTENT_RECOGNIZING = "INTENT_RECOGNIZING"
    INTENT_CLARIFYING = "INTENT_CLARIFYING"           # 意图澄清
    PROFILE_FETCHING = "PROFILE_FETCHING"
    PRODUCT_FETCHING = "PRODUCT_FETCHING"             # 商品信息获取与召回
    SCRIPT_GENERATING = "SCRIPT_GENERATING"
    QUALITY_CHECKING = "QUALITY_CHECKING"
    COMPLETED = "COMPLETED"

    # 多轮对话扩展状态
    CONTENT_LOCATING = "CONTENT_LOCATING"             # 定位历史内容
    SCRIPT_MODIFYING = "SCRIPT_MODIFYING"             # 话术修改

    # 异常状态
    REGENERATING = "REGENERATING"                     # 质量不通过重新生成
    ERROR = "ERROR"
    DEGRADED = "DEGRADED"                             # 降级状态


# ======================================================================
#  Guard Functions - 纯函数, 可独立测试
# ======================================================================

def guard_high_confidence(c: Dict[str, Any]) -> bool:
    """置信度 >= 0.7 → 进入画像获取"""
    return c.get("confidence", 0) >= 0.7


def guard_low_confidence(c: Dict[str, Any]) -> bool:
    """置信度 < 0.5 → 需要用户澄清"""
    return c.get("confidence", 0) < 0.5


def guard_user_clarified(c: Dict[str, Any]) -> bool:
    """用户已完成澄清"""
    return c.get("user_clarified", False)


def guard_is_modification(c: Dict[str, Any]) -> bool:
    """意图为话术修改"""
    return c.get("intent") == "script_modification"


def guard_quality_passed(c: Dict[str, Any]) -> bool:
    """质量校验通过"""
    return c.get("quality_passed", False)


def guard_quality_failed_can_retry(c: Dict[str, Any]) -> bool:
    """质量校验未通过且可重试"""
    return (
        not c.get("quality_passed", True)
        and c.get("retry_count", 0) < c.get("max_retries", 3)
    )


def guard_retry_exhausted(c: Dict[str, Any]) -> bool:
    """重试次数已用完 → 降级"""
    return c.get("retry_count", 0) >= c.get("max_retries", 3)


# ======================================================================
#  Transition & StateContext
# ======================================================================

@dataclass
class Transition:
    """状态转移规则"""
    from_state: WorkflowState
    to_state: WorkflowState
    guard: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""

    def matches(self, conditions: Dict[str, Any]) -> bool:
        """检测此转移是否在给定条件下触发"""
        if self.guard is None:
            return True
        return self.guard(conditions)


@dataclass
class StateContext:
    """状态机上下文 - 记录当前执行状态"""
    current_state: WorkflowState = WorkflowState.INIT
    state_history: List[WorkflowState] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    error_message: str = ""
    state_data: Dict[str, Any] = field(default_factory=dict)


class StateMachine:
    """
    状态机驱动器
    管理工作流状态流转，支持条件分支、重试、降级
    """

    def __init__(self):
        self._transitions: Dict[WorkflowState, List[Transition]] = {}
        self._register_transitions()

    def _register_transitions(self):
        """注册所有合法的状态转移"""
        transitions = [
            # 主流程 (无条件转移)
            Transition(
                WorkflowState.INIT,
                WorkflowState.CONTEXT_LOADING,
                description="初始化 → 加载上下文",
            ),
            Transition(
                WorkflowState.CONTEXT_LOADING,
                WorkflowState.INTENT_RECOGNIZING,
                description="上下文加载完成 → 意图识别",
            ),

            # 意图识别分支 (有条件, guard 函数决定走向)
            # 注意: 有 guard 的排在前面, 优先匹配; 多个 guard 按注册顺序评估
            Transition(
                WorkflowState.INTENT_RECOGNIZING,
                WorkflowState.CONTENT_LOCATING,
                guard=guard_is_modification,
                description="修改类意图 → 定位历史内容",
            ),
            Transition(
                WorkflowState.INTENT_RECOGNIZING,
                WorkflowState.INTENT_CLARIFYING,
                guard=guard_low_confidence,
                description="置信度 < 0.5 → 请求用户澄清",
            ),
            Transition(
                WorkflowState.INTENT_RECOGNIZING,
                WorkflowState.PROFILE_FETCHING,
                guard=guard_high_confidence,
                description="置信度 >= 0.7 → 获取达人画像",
            ),

            # 澄清后重新识别
            Transition(
                WorkflowState.INTENT_CLARIFYING,
                WorkflowState.INTENT_RECOGNIZING,
                guard=guard_user_clarified,
                description="用户已澄清 → 重新识别意图",
            ),

            # 修改类流程
            Transition(
                WorkflowState.CONTENT_LOCATING,
                WorkflowState.SCRIPT_MODIFYING,
                description="内容定位完成 → 修改话术",
            ),
            Transition(
                WorkflowState.SCRIPT_MODIFYING,
                WorkflowState.QUALITY_CHECKING,
                description="话术修改完成 → 质量检查",
            ),

            # 生成流程
            Transition(
                WorkflowState.PROFILE_FETCHING,
                WorkflowState.PRODUCT_FETCHING,
                description="画像获取完成 → 获取商品信息",
            ),
            Transition(
                WorkflowState.PRODUCT_FETCHING,
                WorkflowState.SCRIPT_GENERATING,
                description="商品信息获取完成 → 生成话术",
            ),
            Transition(
                WorkflowState.SCRIPT_GENERATING,
                WorkflowState.QUALITY_CHECKING,
                description="话术生成完成 → 质量检查",
            ),

            # 质量校验分支
            Transition(
                WorkflowState.QUALITY_CHECKING,
                WorkflowState.COMPLETED,
                guard=guard_quality_passed,
                description="质量通过 → 完成",
            ),
            Transition(
                WorkflowState.QUALITY_CHECKING,
                WorkflowState.REGENERATING,
                guard=guard_quality_failed_can_retry,
                description="质量不通过且可重试 → 重新生成",
            ),
            Transition(
                WorkflowState.QUALITY_CHECKING,
                WorkflowState.DEGRADED,
                guard=guard_retry_exhausted,
                description="重试次数用完 → 降级输出",
            ),

            # 重试 → 回到生成
            Transition(
                WorkflowState.REGENERATING,
                WorkflowState.SCRIPT_GENERATING,
                description="重新生成",
            ),
        ]

        for t in transitions:
            self._transitions.setdefault(t.from_state, []).append(t)

    def get_next_state(self, ctx: StateContext,
                       conditions: Dict[str, Any]) -> WorkflowState:
        """
        根据当前状态和条件，确定下一个状态

        Args:
            ctx: 状态机上下文
            conditions: 条件字典, 如 {"confidence": 0.85, "quality_passed": True}

        Returns:
            下一个状态
        """
        available = self._transitions.get(ctx.current_state, [])

        for transition in available:
            if transition.matches(conditions):
                return transition.to_state

        logger.warning(
            f"No valid transition from {ctx.current_state} "
            f"with conditions {conditions}"
        )
        return WorkflowState.ERROR

    def transition(self, ctx: StateContext,
                   conditions: Dict[str, Any]) -> WorkflowState:
        """执行状态转移"""
        old_state = ctx.current_state
        new_state = self.get_next_state(ctx, conditions)

        ctx.state_history.append(old_state)
        ctx.current_state = new_state

        logger.info(f"State transition: {old_state.value} -> {new_state.value}")
        return new_state

    def get_valid_transitions(self, state: WorkflowState) -> List[Transition]:
        """获取某状态的所有可能转移 (用于调试/可视化)"""
        return self._transitions.get(state, [])
