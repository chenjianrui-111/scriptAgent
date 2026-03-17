"""
ScriptAgent协议层

支持多种标准化协议，实现Agent能力的跨系统、跨语言互通：
- MCP (Model Context Protocol): 暴露工具、资源和提示词
- A2A (Agent-to-Agent): Agent间标准化通信
"""

from .mcp_server import (
    MCPServer,
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPToolCall,
    MCPToolResult,
    create_scriptagent_mcp_server,
)

from .a2a_protocol import (
    A2AMessage,
    A2AConversation,
    A2AAgent,
    A2ARouter,
    A2AMessageType,
    A2AAgentRole,
    ScriptAgentA2AAdapter,
    create_a2a_router_with_scriptagent,
)

__all__ = [
    # MCP
    "MCPServer",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPToolCall",
    "MCPToolResult",
    "create_scriptagent_mcp_server",
    # A2A
    "A2AMessage",
    "A2AConversation",
    "A2AAgent",
    "A2ARouter",
    "A2AMessageType",
    "A2AAgentRole",
    "ScriptAgentA2AAdapter",
    "create_a2a_router_with_scriptagent",
]
