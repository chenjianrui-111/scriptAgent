"""
Agent 抽象基类 - 所有子 Agent 的统一接口

提供:
  - 统一的消息处理协议 (__call__ → process)
  - 生命周期管理 (startup / shutdown)
  - 错误处理与降级
"""

import logging
from abc import ABC, abstractmethod

from script_agent.models.message import AgentMessage

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Agent 抽象基类

    所有子 Agent 必须实现 process 方法。
    通过 __call__ 调用时自动包含错误处理。
    """

    def __init__(self, name: str):
        self.name = name
        self._initialized = False
        self.logger = logging.getLogger(f"agent.{name}")

    async def __call__(self, message: AgentMessage) -> AgentMessage:
        """统一调用入口 — 包含错误处理"""
        try:
            return await self.process(message)
        except Exception as e:
            self.logger.error(f"[{self.name}] Processing failed: {e}", exc_info=True)
            return message.create_error(
                error_code=f"{self.name}_error",
                error_msg=str(e),
            )

    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        处理消息 — 子类必须实现

        Args:
            message: 输入消息

        Returns:
            处理后的响应消息
        """
        ...

    async def startup(self):
        """Agent 启动钩子 — 子类可覆盖，用于加载模型/建立连接等"""
        self._initialized = True
        self.logger.info(f"[{self.name}] Started")

    async def shutdown(self):
        """Agent 关闭钩子 — 子类可覆盖，用于释放资源"""
        self._initialized = False
        self.logger.info(f"[{self.name}] Shut down")
