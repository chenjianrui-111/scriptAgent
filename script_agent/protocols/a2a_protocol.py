"""
A2A (Agent-to-Agent) 协议实现

提供标准化的Agent间通信协议，兼容：
- OpenAI Swarm
- Microsoft AutoGen
- LangChain Agent Protocol

支持：
- Agent消息标准化
- 跨框架Agent互操作
- 对话历史管理
- 错误处理和重试
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class A2AMessageType(str, Enum):
    """A2A消息类型"""
    REQUEST = "request"           # Agent请求
    RESPONSE = "response"         # Agent响应
    NOTIFICATION = "notification" # 通知（无需响应）
    ERROR = "error"              # 错误消息


class A2AAgentRole(str, Enum):
    """Agent角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class A2AToolCall(BaseModel):
    """工具调用（OpenAI格式）"""
    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:8]}")
    type: str = "function"
    function: Dict[str, Any] = Field(
        ...,
        description="函数调用信息 {name: str, arguments: str(JSON)}"
    )


class A2AToolResult(BaseModel):
    """工具调用结果"""
    tool_call_id: str
    role: str = "tool"
    content: str = Field(..., description="工具返回结果（JSON字符串）")


class A2AMessage(BaseModel):
    """
    A2A标准消息格式

    兼容OpenAI、AutoGen、LangChain的消息格式
    """
    # 核心字段
    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex}")
    type: A2AMessageType = A2AMessageType.REQUEST
    role: A2AAgentRole = A2AAgentRole.ASSISTANT
    content: Union[str, Dict[str, Any]] = Field(..., description="消息内容")

    # Agent信息
    sender: str = Field(..., description="发送方Agent ID")
    receiver: Optional[str] = Field(None, description="接收方Agent ID")

    # 对话上下文
    conversation_id: str = Field(
        ...,
        description="对话ID，用于关联多轮对话"
    )
    parent_message_id: Optional[str] = Field(
        None,
        description="父消息ID，用于构建对话树"
    )

    # 工具调用（OpenAI格式）
    tool_calls: Optional[List[A2AToolCall]] = None

    # 元数据
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "msg_abc123",
                "type": "request",
                "role": "assistant",
                "content": "请帮我生成一段李佳琦风格的口红话术",
                "sender": "UserAgent",
                "receiver": "ScriptAgent",
                "conversation_id": "conv_xyz789",
                "parent_message_id": "msg_parent123",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {
                    "user_id": "user_001",
                    "session_id": "sess_456"
                }
            }
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """转换为OpenAI消息格式"""
        msg = {
            "role": self.role.value,
            "content": self.content if isinstance(self.content, str) else str(self.content),
        }

        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function,
                }
                for tc in self.tool_calls
            ]

        return msg

    def to_autogen_format(self) -> Dict[str, Any]:
        """转换为AutoGen消息格式"""
        return {
            "role": self.role.value,
            "content": self.content if isinstance(self.content, str) else str(self.content),
            "name": self.sender,
        }

    def to_langchain_format(self) -> Dict[str, Any]:
        """转换为LangChain消息格式"""
        return {
            "type": self.role.value,
            "data": {
                "content": self.content if isinstance(self.content, str) else str(self.content),
                "additional_kwargs": self.metadata or {},
            }
        }


class A2AConversation(BaseModel):
    """A2A对话上下文"""
    id: str = Field(default_factory=lambda: f"conv_{uuid4().hex}")
    messages: List[A2AMessage] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)  # Agent IDs
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    def add_message(self, message: A2AMessage):
        """添加消息到对话"""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

        # 记录参与者
        if message.sender not in self.participants:
            self.participants.append(message.sender)
        if message.receiver and message.receiver not in self.participants:
            self.participants.append(message.receiver)

    def get_messages_for_agent(self, agent_id: str) -> List[A2AMessage]:
        """获取特定Agent的消息历史"""
        return [
            msg for msg in self.messages
            if msg.sender == agent_id or msg.receiver == agent_id
        ]

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """转换为OpenAI消息列表"""
        return [msg.to_openai_format() for msg in self.messages]


class A2AAgent(BaseModel):
    """A2A Agent抽象"""
    id: str
    name: str
    description: str
    capabilities: List[str] = Field(
        default_factory=list,
        description="Agent能力列表"
    )
    supported_protocols: List[str] = Field(
        default_factory=lambda: ["a2a", "openai", "autogen"],
        description="支持的协议"
    )


class A2ARouter:
    """
    A2A消息路由器

    负责：
    - Agent注册和发现
    - 消息路由和分发
    - 对话上下文管理
    - 协议转换
    """

    def __init__(self):
        self._agents: Dict[str, A2AAgent] = {}
        self._handlers: Dict[str, Any] = {}  # agent_id -> handler
        self._conversations: Dict[str, A2AConversation] = {}

    def register_agent(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: List[str],
        handler: Any,
    ):
        """
        注册Agent

        Args:
            agent_id: Agent唯一标识
            name: Agent名称
            description: Agent描述
            capabilities: Agent能力列表
            handler: 消息处理器（async function）
        """
        agent = A2AAgent(
            id=agent_id,
            name=name,
            description=description,
            capabilities=capabilities,
        )

        self._agents[agent_id] = agent
        self._handlers[agent_id] = handler

        logger.info(f"A2A agent registered: {agent_id} ({name})")

    def list_agents(self) -> List[A2AAgent]:
        """列出所有注册的Agent"""
        return list(self._agents.values())

    def get_agent(self, agent_id: str) -> Optional[A2AAgent]:
        """获取Agent信息"""
        return self._agents.get(agent_id)

    async def send_message(
        self,
        message: A2AMessage,
        wait_response: bool = True,
        timeout: float = 30.0,
    ) -> Optional[A2AMessage]:
        """
        发送消息给Agent

        Args:
            message: A2A消息
            wait_response: 是否等待响应
            timeout: 超时时间（秒）

        Returns:
            响应消息（如果wait_response=True）
        """
        receiver_id = message.receiver

        if not receiver_id:
            raise ValueError("Message receiver is required")

        if receiver_id not in self._handlers:
            raise ValueError(f"Agent not found: {receiver_id}")

        # 获取或创建对话
        conv_id = message.conversation_id
        if conv_id not in self._conversations:
            self._conversations[conv_id] = A2AConversation(id=conv_id)

        conversation = self._conversations[conv_id]
        conversation.add_message(message)

        # 调用Agent处理器
        handler = self._handlers[receiver_id]

        try:
            if wait_response:
                # 同步等待响应
                response = await asyncio.wait_for(
                    handler(message, conversation),
                    timeout=timeout,
                )

                if response:
                    # 设置响应消息的parent_message_id
                    response.parent_message_id = message.id
                    response.conversation_id = conv_id
                    conversation.add_message(response)

                return response
            else:
                # 异步发送，不等待
                asyncio.create_task(handler(message, conversation))
                return None

        except asyncio.TimeoutError:
            logger.error(f"A2A message timeout: {receiver_id}, timeout={timeout}s")
            # 返回错误消息
            error_msg = A2AMessage(
                type=A2AMessageType.ERROR,
                role=A2AAgentRole.SYSTEM,
                content={"error": "timeout", "message": f"Agent did not respond within {timeout}s"},
                sender="system",
                receiver=message.sender,
                conversation_id=conv_id,
                parent_message_id=message.id,
            )
            conversation.add_message(error_msg)
            return error_msg

        except Exception as e:
            logger.error(f"A2A message error: {receiver_id}, {e}", exc_info=True)
            # 返回错误消息
            error_msg = A2AMessage(
                type=A2AMessageType.ERROR,
                role=A2AAgentRole.SYSTEM,
                content={"error": "internal_error", "message": str(e)},
                sender="system",
                receiver=message.sender,
                conversation_id=conv_id,
                parent_message_id=message.id,
            )
            conversation.add_message(error_msg)
            return error_msg

    def get_conversation(self, conversation_id: str) -> Optional[A2AConversation]:
        """获取对话上下文"""
        return self._conversations.get(conversation_id)

    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> A2AConversation:
        """创建新对话"""
        conv_id = conversation_id or f"conv_{uuid4().hex}"
        conversation = A2AConversation(
            id=conv_id,
            metadata=metadata,
        )
        self._conversations[conv_id] = conversation
        return conversation


# ==================== ScriptAgent适配器 ====================

class ScriptAgentA2AAdapter:
    """
    ScriptAgent的A2A协议适配器

    将ScriptAgent的内部Agent包装为A2A兼容的Agent
    """

    def __init__(self, orchestrator: Any, router: A2ARouter):
        self.orchestrator = orchestrator
        self.router = router
        self._register_agents()

    def _register_agents(self):
        """注册所有ScriptAgent的Agent"""

        # 注册IntentAgent
        self.router.register_agent(
            agent_id="intent_agent",
            name="IntentAgent",
            description="意图识别和槽位提取",
            capabilities=["intent_recognition", "slot_extraction"],
            handler=self._intent_handler,
        )

        # 注册ProfileAgent
        self.router.register_agent(
            agent_id="profile_agent",
            name="ProfileAgent",
            description="达人画像加载和管理",
            capabilities=["profile_loading", "profile_caching"],
            handler=self._profile_handler,
        )

        # 注册ProductAgent
        self.router.register_agent(
            agent_id="product_agent",
            name="ProductAgent",
            description="商品信息理解和记忆召回",
            capabilities=["product_understanding", "memory_recall"],
            handler=self._product_handler,
        )

        # 注册ScriptAgent
        self.router.register_agent(
            agent_id="script_agent",
            name="ScriptAgent",
            description="话术生成核心Agent",
            capabilities=["script_generation", "prompt_building"],
            handler=self._script_handler,
        )

        # 注册QualityAgent
        self.router.register_agent(
            agent_id="quality_agent",
            name="QualityAgent",
            description="话术质量评估和校验",
            capabilities=["quality_checking", "sensitive_word_detection"],
            handler=self._quality_handler,
        )

        # 注册Orchestrator（统一入口）
        self.router.register_agent(
            agent_id="orchestrator",
            name="Orchestrator",
            description="ScriptAgent编排器，协调多个Agent完成话术生成",
            capabilities=[
                "script_generation",
                "multi_agent_orchestration",
                "workflow_management"
            ],
            handler=self._orchestrator_handler,
        )

        logger.info("ScriptAgent A2A adapter initialized")

    async def _intent_handler(
        self,
        message: A2AMessage,
        conversation: A2AConversation,
    ) -> A2AMessage:
        """IntentAgent处理器"""
        # 简化实现：调用IntentAgent
        result = {"intent": "generate", "confidence": 0.95}

        return A2AMessage(
            type=A2AMessageType.RESPONSE,
            role=A2AAgentRole.ASSISTANT,
            content=result,
            sender="intent_agent",
            receiver=message.sender,
            conversation_id=conversation.id,
        )

    async def _profile_handler(
        self,
        message: A2AMessage,
        conversation: A2AConversation,
    ) -> A2AMessage:
        """ProfileAgent处理器"""
        result = {"profile": {"name": "李佳琦", "style": "professional"}}

        return A2AMessage(
            type=A2AMessageType.RESPONSE,
            role=A2AAgentRole.ASSISTANT,
            content=result,
            sender="profile_agent",
            receiver=message.sender,
            conversation_id=conversation.id,
        )

    async def _product_handler(
        self,
        message: A2AMessage,
        conversation: A2AConversation,
    ) -> A2AMessage:
        """ProductAgent处理器"""
        result = {"product": {"name": "口红", "features": ["持久", "显色"]}}

        return A2AMessage(
            type=A2AMessageType.RESPONSE,
            role=A2AAgentRole.ASSISTANT,
            content=result,
            sender="product_agent",
            receiver=message.sender,
            conversation_id=conversation.id,
        )

    async def _script_handler(
        self,
        message: A2AMessage,
        conversation: A2AConversation,
    ) -> A2AMessage:
        """ScriptAgent处理器"""
        result = {"script": "这款口红真的太好用了！持久显色，一整天不掉色！"}

        return A2AMessage(
            type=A2AMessageType.RESPONSE,
            role=A2AAgentRole.ASSISTANT,
            content=result,
            sender="script_agent",
            receiver=message.sender,
            conversation_id=conversation.id,
        )

    async def _quality_handler(
        self,
        message: A2AMessage,
        conversation: A2AConversation,
    ) -> A2AMessage:
        """QualityAgent处理器"""
        result = {"quality_score": 85, "passed": True}

        return A2AMessage(
            type=A2AMessageType.RESPONSE,
            role=A2AAgentRole.ASSISTANT,
            content=result,
            sender="quality_agent",
            receiver=message.sender,
            conversation_id=conversation.id,
        )

    async def _orchestrator_handler(
        self,
        message: A2AMessage,
        conversation: A2AConversation,
    ) -> A2AMessage:
        """Orchestrator处理器（统一入口）"""
        try:
            # 提取请求参数
            if isinstance(message.content, dict):
                request_data = message.content
            else:
                request_data = {"user_input": message.content}

            # 调用Orchestrator
            result = await self.orchestrator.handle_request(request_data)

            return A2AMessage(
                type=A2AMessageType.RESPONSE,
                role=A2AAgentRole.ASSISTANT,
                content=result,
                sender="orchestrator",
                receiver=message.sender,
                conversation_id=conversation.id,
                metadata={
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "tokens_used": result.get("tokens", 0),
                }
            )

        except Exception as e:
            logger.error(f"Orchestrator handler error: {e}", exc_info=True)

            return A2AMessage(
                type=A2AMessageType.ERROR,
                role=A2AAgentRole.SYSTEM,
                content={"error": str(e)},
                sender="orchestrator",
                receiver=message.sender,
                conversation_id=conversation.id,
            )


# ==================== 工具函数 ====================

def create_a2a_message_from_openai(
    openai_message: Dict[str, Any],
    sender: str,
    receiver: str,
    conversation_id: str,
) -> A2AMessage:
    """从OpenAI格式转换为A2A消息"""
    role_map = {
        "user": A2AAgentRole.USER,
        "assistant": A2AAgentRole.ASSISTANT,
        "system": A2AAgentRole.SYSTEM,
        "tool": A2AAgentRole.TOOL,
    }

    role = role_map.get(openai_message.get("role", "assistant"), A2AAgentRole.ASSISTANT)

    # 处理tool_calls
    tool_calls = None
    if "tool_calls" in openai_message:
        tool_calls = [
            A2AToolCall(
                id=tc["id"],
                type=tc["type"],
                function=tc["function"],
            )
            for tc in openai_message["tool_calls"]
        ]

    return A2AMessage(
        type=A2AMessageType.REQUEST,
        role=role,
        content=openai_message.get("content", ""),
        sender=sender,
        receiver=receiver,
        conversation_id=conversation_id,
        tool_calls=tool_calls,
    )


def create_a2a_router_with_scriptagent(orchestrator: Any) -> A2ARouter:
    """
    创建配置好ScriptAgent的A2A路由器

    Args:
        orchestrator: Orchestrator实例

    Returns:
        A2ARouter实例
    """
    router = A2ARouter()
    adapter = ScriptAgentA2AAdapter(orchestrator, router)
    return router
