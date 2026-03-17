"""
A2A (Agent-to-Agent) 协议使用示例

演示如何使用A2A协议实现Agent间标准化通信。
"""

import asyncio
import json
import logging
from datetime import datetime

from script_agent.protocols.a2a_protocol import (
    A2AMessage,
    A2ARouter,
    A2AMessageType,
    A2AAgentRole,
    create_a2a_router_with_scriptagent,
    create_a2a_message_from_openai,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 模拟组件 ====================

class MockOrchestrator:
    """模拟Orchestrator"""
    async def handle_request(self, payload):
        await asyncio.sleep(0.5)
        return {
            "script": "生成的话术内容",
            "quality_score": 85,
            "tokens": 1200,
            "processing_time_ms": 500,
        }


# ==================== 基础示例 ====================

async def basic_a2a_example():
    """基础A2A协议示例"""
    logger.info("=== Basic A2A Protocol Example ===\n")

    # 1. 创建A2A路由器
    router = A2ARouter()

    # 2. 注册一个简单的Agent
    async def echo_handler(message, conversation):
        """Echo Agent：回显消息"""
        return A2AMessage(
            type=A2AMessageType.RESPONSE,
            role=A2AAgentRole.ASSISTANT,
            content=f"Echo: {message.content}",
            sender="echo_agent",
            receiver=message.sender,
            conversation_id=conversation.id,
        )

    router.register_agent(
        agent_id="echo_agent",
        name="EchoAgent",
        description="简单的回显Agent",
        capabilities=["echo"],
        handler=echo_handler,
    )

    # 3. 列出所有Agent
    agents = router.list_agents()
    logger.info(f"Registered Agents ({len(agents)}):")
    for agent in agents:
        logger.info(f"  - {agent.id}: {agent.description}")
    logger.info("")

    # 4. 创建对话
    conversation = router.create_conversation()
    logger.info(f"Created conversation: {conversation.id}\n")

    # 5. 发送消息
    logger.info("发送消息到echo_agent...")
    message = A2AMessage(
        type=A2AMessageType.REQUEST,
        role=A2AAgentRole.USER,
        content="Hello, World!",
        sender="user",
        receiver="echo_agent",
        conversation_id=conversation.id,
    )

    response = await router.send_message(message)

    logger.info(f"收到响应:")
    logger.info(f"  Sender: {response.sender}")
    logger.info(f"  Content: {response.content}\n")


# ==================== ScriptAgent集成示例 ====================

async def scriptagent_integration_example():
    """ScriptAgent A2A集成示例"""
    logger.info("=== ScriptAgent A2A Integration Example ===\n")

    # 1. 创建配置好的A2A路由器
    orchestrator = MockOrchestrator()
    router = create_a2a_router_with_scriptagent(orchestrator)

    # 2. 列出ScriptAgent的所有Agent
    agents = router.list_agents()
    logger.info(f"ScriptAgent Agents ({len(agents)}):")
    for agent in agents:
        logger.info(f"  - {agent.id}")
        logger.info(f"    Name: {agent.name}")
        logger.info(f"    Description: {agent.description}")
        logger.info(f"    Capabilities: {', '.join(agent.capabilities)}")
    logger.info("")

    # 3. 创建对话
    conversation = router.create_conversation(
        metadata={"user_id": "user_001", "session_type": "demo"}
    )

    # 4. 发送话术生成请求到Orchestrator
    logger.info("发送请求: 生成李佳琦风格的口红话术")
    message = A2AMessage(
        type=A2AMessageType.REQUEST,
        role=A2AAgentRole.USER,
        content={
            "talent_name": "李佳琦",
            "product_name": "YSL口红",
            "style": "professional",
        },
        sender="user_agent",
        receiver="orchestrator",
        conversation_id=conversation.id,
    )

    response = await router.send_message(message, timeout=10.0)

    logger.info(f"\n收到响应:")
    logger.info(f"  Status: {response.type}")
    logger.info(f"  Content: {json.dumps(response.content, ensure_ascii=False, indent=2)}")
    logger.info(f"  Metadata: {response.metadata}\n")


# ==================== 多Agent协作示例 ====================

async def multi_agent_collaboration_example():
    """多Agent协作示例"""
    logger.info("=== Multi-Agent Collaboration Example ===\n")

    router = A2ARouter()

    # 注册多个Agent
    async def translator_handler(message, conversation):
        """翻译Agent"""
        await asyncio.sleep(0.2)
        return A2AMessage(
            type=A2AMessageType.RESPONSE,
            role=A2AAgentRole.ASSISTANT,
            content=f"Translated: {message.content}",
            sender="translator",
            receiver=message.sender,
            conversation_id=conversation.id,
        )

    async def summarizer_handler(message, conversation):
        """摘要Agent"""
        await asyncio.sleep(0.2)
        return A2AMessage(
            type=A2AMessageType.RESPONSE,
            role=A2AAgentRole.ASSISTANT,
            content=f"Summary: {message.content[:50]}...",
            sender="summarizer",
            receiver=message.sender,
            conversation_id=conversation.id,
        )

    router.register_agent("translator", "Translator", "翻译Agent", ["translate"], translator_handler)
    router.register_agent("summarizer", "Summarizer", "摘要Agent", ["summarize"], summarizer_handler)

    # 协作流程
    conversation = router.create_conversation()

    logger.info("场景: 用户输入 → 翻译Agent → 摘要Agent\n")

    # Step 1: 发送到翻译Agent
    logger.info("Step 1: 发送到翻译Agent")
    msg1 = A2AMessage(
        type=A2AMessageType.REQUEST,
        role=A2AAgentRole.USER,
        content="这是一段很长的中文文本...",
        sender="user",
        receiver="translator",
        conversation_id=conversation.id,
    )
    resp1 = await router.send_message(msg1)
    logger.info(f"  翻译结果: {resp1.content}\n")

    # Step 2: 发送到摘要Agent
    logger.info("Step 2: 发送到摘要Agent")
    msg2 = A2AMessage(
        type=A2AMessageType.REQUEST,
        role=A2AAgentRole.ASSISTANT,
        content=resp1.content,
        sender="translator",
        receiver="summarizer",
        conversation_id=conversation.id,
        parent_message_id=resp1.id,
    )
    resp2 = await router.send_message(msg2)
    logger.info(f"  摘要结果: {resp2.content}\n")

    # 查看对话历史
    conv = router.get_conversation(conversation.id)
    logger.info(f"对话历史 ({len(conv.messages)} 条消息):")
    for i, msg in enumerate(conv.messages, 1):
        logger.info(f"  {i}. {msg.sender} → {msg.receiver}: {str(msg.content)[:30]}...")
    logger.info("")


# ==================== OpenAI格式兼容示例 ====================

async def openai_compatibility_example():
    """OpenAI格式兼容示例"""
    logger.info("=== OpenAI Compatibility Example ===\n")

    # 1. OpenAI格式消息
    openai_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "生成一段口红话术"},
        {"role": "assistant", "content": "这款口红..."},
    ]

    logger.info("OpenAI格式消息:")
    logger.info(json.dumps(openai_messages, ensure_ascii=False, indent=2))
    logger.info("")

    # 2. 转换为A2A格式
    conversation_id = "conv_openai_demo"
    a2a_messages = [
        create_a2a_message_from_openai(
            msg,
            sender="openai_client",
            receiver="scriptagent",
            conversation_id=conversation_id,
        )
        for msg in openai_messages
    ]

    logger.info("转换为A2A格式:")
    for msg in a2a_messages:
        logger.info(f"  {msg.role}: {msg.content[:30]}...")
    logger.info("")

    # 3. A2A消息转回OpenAI格式
    openai_converted = [msg.to_openai_format() for msg in a2a_messages]

    logger.info("转回OpenAI格式:")
    logger.info(json.dumps(openai_converted, ensure_ascii=False, indent=2))
    logger.info("")


# ==================== 错误处理示例 ====================

async def error_handling_example():
    """错误处理示例"""
    logger.info("=== Error Handling Example ===\n")

    router = A2ARouter()

    # 注册会抛出异常的Agent
    async def faulty_handler(message, conversation):
        """有问题的Agent"""
        raise ValueError("模拟Agent内部错误")

    router.register_agent("faulty", "FaultyAgent", "会出错的Agent", [], faulty_handler)

    conversation = router.create_conversation()

    # 发送消息
    logger.info("发送消息到faulty_agent（会出错）")
    message = A2AMessage(
        type=A2AMessageType.REQUEST,
        role=A2AAgentRole.USER,
        content="Test error handling",
        sender="user",
        receiver="faulty",
        conversation_id=conversation.id,
    )

    response = await router.send_message(message)

    # 检查错误响应
    logger.info(f"\n错误响应:")
    logger.info(f"  Type: {response.type}")
    logger.info(f"  Content: {response.content}")
    logger.info("")

    # 测试超时
    async def slow_handler(message, conversation):
        """慢Agent"""
        await asyncio.sleep(10)
        return A2AMessage(
            type=A2AMessageType.RESPONSE,
            role=A2AAgentRole.ASSISTANT,
            content="Finally done!",
            sender="slow",
            receiver=message.sender,
            conversation_id=conversation.id,
        )

    router.register_agent("slow", "SlowAgent", "很慢的Agent", [], slow_handler)

    logger.info("发送消息到slow_agent（会超时）")
    message = A2AMessage(
        type=A2AMessageType.REQUEST,
        role=A2AAgentRole.USER,
        content="Test timeout",
        sender="user",
        receiver="slow",
        conversation_id=conversation.id,
    )

    response = await router.send_message(message, timeout=1.0)

    logger.info(f"\n超时响应:")
    logger.info(f"  Type: {response.type}")
    logger.info(f"  Content: {response.content}\n")


# ==================== 对比不同框架 ====================

async def framework_comparison():
    """对比A2A在不同框架中的应用"""
    logger.info("=== Framework Comparison ===\n")

    logger.info("【OpenAI Swarm】")
    logger.info("  特点: 轻量级，强调Agent切换")
    logger.info("  消息格式: 类OpenAI chat格式")
    logger.info("  适用: 简单多Agent协作\n")

    logger.info("【Microsoft AutoGen】")
    logger.info("  特点: 会话式，强调代码执行")
    logger.info("  消息格式: {role, content, name}")
    logger.info("  适用: 代码生成、自动化任务\n")

    logger.info("【LangChain Agent】")
    logger.info("  特点: 工具调用为核心")
    logger.info("  消息格式: BaseMessage子类")
    logger.info("  适用: RAG、工具链\n")

    logger.info("【ScriptAgent A2A】")
    logger.info("  特点: 统一以上三种格式，支持互转")
    logger.info("  消息格式: A2AMessage（标准化）")
    logger.info("  适用: 跨框架Agent互操作\n")

    logger.info("类比: A2A之于Agent ≈ HTTP之于Web服务\n")


# ==================== 主函数 ====================

async def main():
    """主函数"""
    # 运行所有示例
    await basic_a2a_example()
    await scriptagent_integration_example()
    await multi_agent_collaboration_example()
    await openai_compatibility_example()
    await error_handling_example()
    await framework_comparison()

    logger.info("=== All Examples Completed ===")


if __name__ == "__main__":
    asyncio.run(main())
