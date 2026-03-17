"""
MCP (Model Context Protocol) Server使用示例

演示如何使用MCP协议暴露ScriptAgent的能力。
"""

import asyncio
import json
import logging

from script_agent.protocols.mcp_server import (
    MCPServer,
    MCPToolCall,
    MCPResourceContent,
    create_scriptagent_mcp_server,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 模拟组件 ====================

class MockOrchestrator:
    """模拟Orchestrator"""
    async def handle_request(self, payload):
        await asyncio.sleep(0.5)
        return {
            "script": f"【{payload['talent_name']}风格】{payload['product_name']}真的太好用了！",
            "quality_score": 85,
            "intent": "generate",
            "tokens": 1200,
        }


class MockDomainRepository:
    """模拟数据仓库"""
    async def get_influencer_by_id(self, talent_id: str):
        return {
            "id": talent_id,
            "name": "李佳琦",
            "style": "专业热情",
            "tone": "亲切自然",
            "signature_phrases": ["OMG", "买它买它", "所有女生"],
        }

    async def get_product_by_id(self, product_id: str):
        return {
            "id": product_id,
            "name": "YSL星辰口红",
            "category": "美妆/口红",
            "features": ["持久", "显色", "滋润"],
            "selling_points": ["明星同款", "限量色号", "高性价比"],
        }


# ==================== 基础示例 ====================

async def basic_mcp_example():
    """基础MCP服务器示例"""
    logger.info("=== Basic MCP Server Example ===\n")

    # 1. 创建MCP服务器
    orchestrator = MockOrchestrator()
    domain_repo = MockDomainRepository()

    server = create_scriptagent_mcp_server(
        orchestrator=orchestrator,
        domain_repo=domain_repo,
    )

    # 2. 查看服务器信息
    info = server.get_server_info()
    logger.info(f"Server Info:\n{json.dumps(info, indent=2)}\n")

    # 3. 列出可用工具
    tools = server.list_tools()
    logger.info(f"Available Tools ({len(tools)}):")
    for tool in tools:
        logger.info(f"  - {tool.name}: {tool.description}")
    logger.info("")

    # 4. 列出可用资源
    resources = server.list_resources()
    logger.info(f"Available Resources ({len(resources)}):")
    for resource in resources:
        logger.info(f"  - {resource.uri}: {resource.description}")
    logger.info("")

    # 5. 列出可用提示词
    prompts = server.list_prompts()
    logger.info(f"Available Prompts ({len(prompts)}):")
    for prompt in prompts:
        logger.info(f"  - {prompt.name}: {prompt.description}")
    logger.info("")


# ==================== 工具调用示例 ====================

async def tool_call_example():
    """MCP工具调用示例"""
    logger.info("=== MCP Tool Call Example ===\n")

    # 创建服务器
    orchestrator = MockOrchestrator()
    domain_repo = MockDomainRepository()
    server = create_scriptagent_mcp_server(orchestrator, domain_repo)

    # 调用generate_script工具
    logger.info("调用工具: generate_script")
    tool_call = MCPToolCall(
        tool="generate_script",
        arguments={
            "talent_name": "李佳琦",
            "product_name": "YSL口红",
            "style": "professional",
            "length": "medium",
        }
    )

    result = await server.call_tool(tool_call)

    logger.info(f"\n工具调用结果:")
    logger.info(f"  Tool: {result.tool}")
    logger.info(f"  Result: {json.dumps(result.result, ensure_ascii=False, indent=2)}")
    logger.info(f"  Metadata: {result.metadata}\n")


# ==================== 资源读取示例 ====================

async def resource_read_example():
    """MCP资源读取示例"""
    logger.info("=== MCP Resource Read Example ===\n")

    # 创建服务器
    orchestrator = MockOrchestrator()
    domain_repo = MockDomainRepository()
    server = create_scriptagent_mcp_server(orchestrator, domain_repo)

    # 读取达人画像资源
    logger.info("读取资源: scriptagent://talent/T001")
    resource = await server.read_resource("scriptagent://talent/T001")

    if resource:
        logger.info(f"\n资源内容:")
        logger.info(f"  URI: {resource.uri}")
        logger.info(f"  Type: {resource.type}")
        logger.info(f"  Content: {json.dumps(resource.content, ensure_ascii=False, indent=2)}\n")
    else:
        logger.info("资源未找到\n")

    # 读取商品资源
    logger.info("读取资源: scriptagent://product/P001")
    resource = await server.read_resource("scriptagent://product/P001")

    if resource:
        logger.info(f"\n资源内容:")
        logger.info(f"  URI: {resource.uri}")
        logger.info(f"  Type: {resource.type}")
        logger.info(f"  Content: {json.dumps(resource.content, ensure_ascii=False, indent=2)}\n")


# ==================== 提示词渲染示例 ====================

async def prompt_render_example():
    """MCP提示词渲染示例"""
    logger.info("=== MCP Prompt Render Example ===\n")

    # 创建服务器
    orchestrator = MockOrchestrator()
    domain_repo = MockDomainRepository()
    server = create_scriptagent_mcp_server(orchestrator, domain_repo)

    # 渲染开场话术提示词
    logger.info("渲染提示词: generate_opening")
    result = server.render_prompt(
        name="generate_opening",
        arguments={
            "talent_name": "李佳琦",
            "product_name": "YSL口红",
            "time_of_day": "晚上8点",
        }
    )

    if result:
        logger.info(f"\n渲染结果:")
        logger.info(f"  Prompt: {result.prompt}")
        logger.info(f"  Rendered:\n{result.rendered}\n")

    # 渲染结尾话术提示词
    logger.info("渲染提示词: generate_closing")
    result = server.render_prompt(
        name="generate_closing",
        arguments={
            "talent_name": "李佳琦",
            "call_to_action": "点击链接立即购买",
        }
    )

    if result:
        logger.info(f"\n渲染结果:")
        logger.info(f"  Prompt: {result.prompt}")
        logger.info(f"  Rendered:\n{result.rendered}\n")


# ==================== 跨语言集成示例 ====================

async def cross_language_integration_example():
    """跨语言Agent集成示例"""
    logger.info("=== Cross-Language Integration Example ===\n")

    logger.info("场景: Node.js前端Agent调用Python后端ScriptAgent\n")

    # 1. 前端Agent（Node.js）通过MCP协议发现能力
    logger.info("步骤1: 前端Agent发现后端能力")
    logger.info("  GET /mcp/tools")
    logger.info("  响应: [generate_script, evaluate_quality]\n")

    # 2. 前端Agent调用后端工具
    logger.info("步骤2: 前端Agent调用generate_script")
    logger.info("  POST /mcp/tools/generate_script")
    logger.info("  Body: {talent_name: '李佳琦', product_name: 'YSL口红'}\n")

    # 模拟调用
    orchestrator = MockOrchestrator()
    domain_repo = MockDomainRepository()
    server = create_scriptagent_mcp_server(orchestrator, domain_repo)

    result = await server.call_tool(MCPToolCall(
        tool="generate_script",
        arguments={
            "talent_name": "李佳琦",
            "product_name": "YSL口红",
        }
    ))

    logger.info(f"步骤3: 后端返回结果")
    logger.info(f"  {json.dumps(result.result, ensure_ascii=False, indent=2)}\n")

    logger.info("结论: MCP协议实现了跨语言Agent无缝集成！\n")


# ==================== 对比传统方式 ====================

async def comparison_example():
    """对比MCP vs 传统API"""
    logger.info("=== MCP vs Traditional API ===\n")

    logger.info("【传统方式】自定义API")
    logger.info("  问题:")
    logger.info("    1. 每个系统定义不同的API格式")
    logger.info("    2. 集成需要写大量适配代码")
    logger.info("    3. 无法自动发现能力")
    logger.info("    4. 缺乏标准化的错误处理\n")

    logger.info("【MCP方式】标准化协议")
    logger.info("  优势:")
    logger.info("    1. 统一的工具、资源、提示词抽象")
    logger.info("    2. 自动发现和调用（类似OpenAPI）")
    logger.info("    3. 跨语言、跨框架互操作")
    logger.info("    4. 符合Claude/OpenAI的生态标准\n")

    logger.info("类比: MCP之于AI Agent ≈ OpenAPI之于REST API\n")


# ==================== 主函数 ====================

async def main():
    """主函数"""
    # 运行所有示例
    await basic_mcp_example()
    await tool_call_example()
    await resource_read_example()
    await prompt_render_example()
    await cross_language_integration_example()
    await comparison_example()

    logger.info("=== All Examples Completed ===")


if __name__ == "__main__":
    asyncio.run(main())
