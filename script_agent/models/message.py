"""
标准化消息协议 - AgentMessage
所有Agent间通信使用统一消息格式，通过trace_id串联全链路
"""

import uuid
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class MessageType(str, Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"
    STREAM = "stream"


class AgentRole(str, Enum):
    """Agent角色"""
    ORCHESTRATOR = "orchestrator"
    INTENT = "intent_recognition"
    PROFILE = "profile"
    PRODUCT = "product"
    SCRIPT_GENERATION = "script_generation"
    QUALITY_CHECK = "quality_check"


@dataclass
class AgentMessage:
    """
    标准化Agent间消息协议
    支持同步/异步/流式通信，trace_id串联全链路
    """
    # 消息标识
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""                # 全链路追踪ID
    parent_message_id: str = ""       # 父消息ID（用于链路追踪）

    # 路由信息
    source: str = ""                  # 发送方Agent
    target: str = ""                  # 接收方Agent
    message_type: MessageType = MessageType.REQUEST

    # 业务数据
    payload: Dict[str, Any] = field(default_factory=dict)

    # 上下文引用
    session_id: str = ""
    tenant_id: str = ""

    # 元数据
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def create_response(self, payload: Dict[str, Any],
                        source: str = "") -> "AgentMessage":
        """创建对当前消息的响应"""
        return AgentMessage(
            trace_id=self.trace_id,
            parent_message_id=self.message_id,
            source=source or self.target,
            target=self.source,
            message_type=MessageType.RESPONSE,
            payload=payload,
            session_id=self.session_id,
            tenant_id=self.tenant_id,
        )

    def create_error(self, error_code: str, error_msg: str) -> "AgentMessage":
        """创建错误响应"""
        return AgentMessage(
            trace_id=self.trace_id,
            parent_message_id=self.message_id,
            source=self.target,
            target=self.source,
            message_type=MessageType.ERROR,
            payload={"error_code": error_code, "error_message": error_msg},
            session_id=self.session_id,
            tenant_id=self.tenant_id,
        )


@dataclass
class IntentResult:
    """意图识别结果"""
    intent: str                         # 意图类型
    confidence: float                   # 置信度
    slots: Dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False   # 是否需要用户澄清
    clarification_question: str = ""    # 澄清问题
    inferred: bool = False              # 是否为推理得出
    reasoning: str = ""                 # 推理过程


@dataclass
class GeneratedScript:
    """生成的话术"""
    script_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    category: str = ""
    scenario: str = ""
    style_keywords: List[str] = field(default_factory=list)
    turn_index: int = 0
    adopted: bool = False               # 是否被用户采纳
    quality_score: float = 0.0
    generation_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityResult:
    """质量校验结果"""
    passed: bool = True
    overall_score: float = 0.0
    sensitive_words: List[str] = field(default_factory=list)
    compliance_issues: List[str] = field(default_factory=list)
    style_consistency: float = 0.0
    suggestions: List[str] = field(default_factory=list)
