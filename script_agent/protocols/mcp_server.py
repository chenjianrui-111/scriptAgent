"""
MCP (Model Context Protocol) Server实现

提供标准化的上下文协议服务，支持：
- Tools: 暴露话术生成、质量评分等能力
- Resources: 提供达人画像、商品数据等资源
- Prompts: 分享话术生成模板

参考：https://modelcontextprotocol.io/
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class MCPResourceType(str, Enum):
    """MCP资源类型"""
    TEXT = "text"
    BLOB = "blob"
    JSON = "json"


class MCPToolType(str, Enum):
    """MCP工具类型"""
    FUNCTION = "function"


class MCPPromptType(str, Enum):
    """MCP提示词类型"""
    TEMPLATE = "template"
    EXAMPLE = "example"


# ==================== MCP消息定义 ====================

class MCPTool(BaseModel):
    """MCP工具定义"""
    name: str
    description: str
    type: MCPToolType = MCPToolType.FUNCTION
    input_schema: Dict[str, Any] = Field(
        ...,
        description="JSON Schema定义输入参数"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "generate_script",
                "description": "生成电商话术",
                "type": "function",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "talent_name": {"type": "string"},
                        "product_name": {"type": "string"},
                        "style": {"type": "string", "enum": ["professional", "casual"]}
                    },
                    "required": ["talent_name", "product_name"]
                }
            }
        }


class MCPToolCall(BaseModel):
    """MCP工具调用"""
    tool: str
    arguments: Dict[str, Any]


class MCPToolResult(BaseModel):
    """MCP工具调用结果"""
    tool: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MCPResource(BaseModel):
    """MCP资源定义"""
    uri: str = Field(..., description="资源URI，如 scriptagent://talent/T001")
    name: str
    description: str
    type: MCPResourceType = MCPResourceType.JSON

    class Config:
        json_schema_extra = {
            "example": {
                "uri": "scriptagent://talent/T001",
                "name": "达人画像 - 李佳琦",
                "description": "李佳琦的直播风格、常用话术等画像数据",
                "type": "json"
            }
        }


class MCPResourceContent(BaseModel):
    """MCP资源内容"""
    uri: str
    type: MCPResourceType
    content: Any  # text/blob/json


class MCPPrompt(BaseModel):
    """MCP提示词定义"""
    name: str
    description: str
    type: MCPPromptType = MCPPromptType.TEMPLATE
    arguments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="模板参数定义"
    )
    template: str = Field(..., description="Prompt模板内容")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "generate_opening",
                "description": "生成直播开场话术",
                "type": "template",
                "arguments": [
                    {"name": "talent_name", "type": "string", "required": True},
                    {"name": "product_name", "type": "string", "required": True}
                ],
                "template": "作为{talent_name}，为{product_name}生成一段开场白..."
            }
        }


class MCPPromptResult(BaseModel):
    """MCP提示词渲染结果"""
    prompt: str
    rendered: str
    metadata: Optional[Dict[str, Any]] = None


# ==================== MCP Server实现 ====================

class MCPServer:
    """
    MCP协议服务器

    实现Model Context Protocol标准，对外暴露ScriptAgent的能力。
    """

    def __init__(
        self,
        server_name: str = "ScriptAgent MCP Server",
        version: str = "1.0.0",
        description: str = "电商话术生成Agent的MCP服务",
    ):
        self.server_name = server_name
        self.version = version
        self.description = description

        # 注册表
        self._tools: Dict[str, MCPTool] = {}
        self._tool_handlers: Dict[str, Callable] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._resource_handlers: Dict[str, Callable] = {}
        self._prompts: Dict[str, MCPPrompt] = {}

        logger.info(f"MCP Server initialized: {server_name} v{version}")

    # ==================== Tool注册与调用 ====================

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
    ):
        """
        注册MCP工具

        Example:
            server.register_tool(
                name="generate_script",
                description="生成电商话术",
                input_schema={
                    "type": "object",
                    "properties": {
                        "talent_name": {"type": "string"},
                        "product_name": {"type": "string"}
                    },
                    "required": ["talent_name", "product_name"]
                },
                handler=generate_script_handler
            )
        """
        tool = MCPTool(
            name=name,
            description=description,
            type=MCPToolType.FUNCTION,
            input_schema=input_schema,
        )

        self._tools[name] = tool
        self._tool_handlers[name] = handler
        logger.info(f"MCP tool registered: {name}")

    def list_tools(self) -> List[MCPTool]:
        """列出所有可用工具"""
        return list(self._tools.values())

    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """
        调用MCP工具

        Args:
            tool_call: 工具调用请求

        Returns:
            工具调用结果
        """
        tool_name = tool_call.tool

        if tool_name not in self._tool_handlers:
            return MCPToolResult(
                tool=tool_name,
                error=f"Tool not found: {tool_name}"
            )

        try:
            # 调用处理器
            handler = self._tool_handlers[tool_name]
            result = await handler(**tool_call.arguments)

            return MCPToolResult(
                tool=tool_name,
                result=result,
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "server": self.server_name,
                }
            )

        except Exception as e:
            logger.error(f"MCP tool call error: {tool_name}, {e}", exc_info=True)
            return MCPToolResult(
                tool=tool_name,
                error=str(e)
            )

    # ==================== Resource注册与获取 ====================

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        resource_type: MCPResourceType,
        handler: Callable,
    ):
        """
        注册MCP资源

        Example:
            server.register_resource(
                uri="scriptagent://talent/{talent_id}",
                name="达人画像",
                description="获取达人的风格画像数据",
                resource_type=MCPResourceType.JSON,
                handler=get_talent_profile_handler
            )
        """
        resource = MCPResource(
            uri=uri,
            name=name,
            description=description,
            type=resource_type,
        )

        self._resources[uri] = resource
        self._resource_handlers[uri] = handler
        logger.info(f"MCP resource registered: {uri}")

    def list_resources(self) -> List[MCPResource]:
        """列出所有可用资源"""
        return list(self._resources.values())

    async def read_resource(self, uri: str) -> Optional[MCPResourceContent]:
        """
        读取MCP资源

        Args:
            uri: 资源URI

        Returns:
            资源内容
        """
        # 匹配URI模式（支持路径参数）
        handler = None
        resource_type = None

        for registered_uri, resource in self._resources.items():
            if self._match_uri(registered_uri, uri):
                handler = self._resource_handlers[registered_uri]
                resource_type = resource.type
                break

        if not handler:
            return None

        try:
            # 解析URI参数
            params = self._extract_uri_params(registered_uri, uri)

            # 调用处理器
            content = await handler(**params)

            return MCPResourceContent(
                uri=uri,
                type=resource_type,
                content=content,
            )

        except Exception as e:
            logger.error(f"MCP resource read error: {uri}, {e}", exc_info=True)
            return None

    # ==================== Prompt注册与渲染 ====================

    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: List[Dict[str, Any]],
        template: str,
    ):
        """
        注册MCP提示词模板

        Example:
            server.register_prompt(
                name="generate_opening",
                description="生成直播开场话术",
                arguments=[
                    {"name": "talent_name", "type": "string", "required": True},
                    {"name": "product_name", "type": "string", "required": True}
                ],
                template="作为{talent_name}，为{product_name}生成一段开场白..."
            )
        """
        prompt = MCPPrompt(
            name=name,
            description=description,
            type=MCPPromptType.TEMPLATE,
            arguments=arguments,
            template=template,
        )

        self._prompts[name] = prompt
        logger.info(f"MCP prompt registered: {name}")

    def list_prompts(self) -> List[MCPPrompt]:
        """列出所有可用提示词"""
        return list(self._prompts.values())

    def render_prompt(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Optional[MCPPromptResult]:
        """
        渲染MCP提示词

        Args:
            name: 提示词名称
            arguments: 模板参数

        Returns:
            渲染结果
        """
        if name not in self._prompts:
            return None

        prompt = self._prompts[name]

        try:
            # 简单模板渲染（实际可用Jinja2）
            rendered = prompt.template.format(**arguments)

            return MCPPromptResult(
                prompt=name,
                rendered=rendered,
                metadata={
                    "arguments": arguments,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        except Exception as e:
            logger.error(f"MCP prompt render error: {name}, {e}", exc_info=True)
            return None

    # ==================== 辅助方法 ====================

    def _match_uri(self, pattern: str, uri: str) -> bool:
        """匹配URI模式"""
        # 简单实现：支持 {param} 占位符
        import re
        regex = re.sub(r'\{[^}]+\}', r'[^/]+', pattern)
        return re.fullmatch(regex, uri) is not None

    def _extract_uri_params(self, pattern: str, uri: str) -> Dict[str, str]:
        """从URI提取参数"""
        import re

        # 提取参数名
        param_names = re.findall(r'\{([^}]+)\}', pattern)

        # 构建正则表达式
        regex = re.sub(r'\{[^}]+\}', r'([^/]+)', pattern)

        # 匹配并提取值
        match = re.fullmatch(regex, uri)
        if not match:
            return {}

        return dict(zip(param_names, match.groups()))

    # ==================== Server Info ====================

    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "name": self.server_name,
            "version": self.version,
            "description": self.description,
            "capabilities": {
                "tools": len(self._tools),
                "resources": len(self._resources),
                "prompts": len(self._prompts),
            },
            "protocol_version": "1.0",
        }


# ==================== 工厂函数 ====================

def create_scriptagent_mcp_server(
    orchestrator: Any,  # Orchestrator实例
    domain_repo: Any,   # DomainDataRepository实例
) -> MCPServer:
    """
    创建ScriptAgent的MCP服务器，注册所有工具、资源和提示词

    Args:
        orchestrator: Orchestrator实例（用于话术生成）
        domain_repo: DomainDataRepository实例（用于数据访问）

    Returns:
        配置好的MCP服务器实例
    """
    server = MCPServer(
        server_name="ScriptAgent MCP Server",
        version="1.0.0",
        description="电商话术生成多Agent系统的MCP服务接口",
    )

    # ==================== 注册Tools ====================

    async def generate_script_handler(
        talent_name: str,
        product_name: str,
        style: str = "professional",
        length: str = "medium",
    ) -> Dict[str, Any]:
        """生成话术工具"""
        # 调用Orchestrator生成话术
        result = await orchestrator.handle_request({
            "talent_name": talent_name,
            "product_name": product_name,
            "style": style,
            "length": length,
        })

        return {
            "script": result.get("script", ""),
            "quality_score": result.get("quality_score", 0),
            "metadata": {
                "intent": result.get("intent", ""),
                "tokens": result.get("tokens", 0),
            }
        }

    server.register_tool(
        name="generate_script",
        description="生成电商直播/短视频话术",
        input_schema={
            "type": "object",
            "properties": {
                "talent_name": {
                    "type": "string",
                    "description": "达人名称"
                },
                "product_name": {
                    "type": "string",
                    "description": "商品名称"
                },
                "style": {
                    "type": "string",
                    "enum": ["professional", "casual", "humorous"],
                    "default": "professional",
                    "description": "话术风格"
                },
                "length": {
                    "type": "string",
                    "enum": ["short", "medium", "long"],
                    "default": "medium",
                    "description": "话术长度"
                }
            },
            "required": ["talent_name", "product_name"]
        },
        handler=generate_script_handler,
    )

    async def evaluate_quality_handler(script: str) -> Dict[str, Any]:
        """质量评估工具"""
        # 这里应该调用QualityAgent
        # 简化实现
        return {
            "overall_score": 85,
            "dimensions": {
                "sensitive_words": 100,
                "compliance": 90,
                "style_consistency": 85,
                "structure": 80,
            },
            "issues": []
        }

    server.register_tool(
        name="evaluate_quality",
        description="评估话术质量",
        input_schema={
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "待评估的话术文本"
                }
            },
            "required": ["script"]
        },
        handler=evaluate_quality_handler,
    )

    # ==================== 注册Resources ====================

    async def get_talent_profile_handler(talent_id: str) -> Dict[str, Any]:
        """获取达人画像"""
        profile = await domain_repo.get_influencer_by_id(talent_id)
        if not profile:
            raise ValueError(f"Talent not found: {talent_id}")

        return {
            "id": profile.get("id"),
            "name": profile.get("name"),
            "style": profile.get("style"),
            "tone": profile.get("tone"),
            "signature_phrases": profile.get("signature_phrases", []),
        }

    server.register_resource(
        uri="scriptagent://talent/{talent_id}",
        name="达人画像",
        description="获取达人的风格画像数据（风格、语气、签名话术等）",
        resource_type=MCPResourceType.JSON,
        handler=get_talent_profile_handler,
    )

    async def get_product_info_handler(product_id: str) -> Dict[str, Any]:
        """获取商品信息"""
        product = await domain_repo.get_product_by_id(product_id)
        if not product:
            raise ValueError(f"Product not found: {product_id}")

        return {
            "id": product.get("id"),
            "name": product.get("name"),
            "category": product.get("category"),
            "features": product.get("features", []),
            "selling_points": product.get("selling_points", []),
        }

    server.register_resource(
        uri="scriptagent://product/{product_id}",
        name="商品信息",
        description="获取商品的详细信息（类目、特征、卖点等）",
        resource_type=MCPResourceType.JSON,
        handler=get_product_info_handler,
    )

    # ==================== 注册Prompts ====================

    server.register_prompt(
        name="generate_opening",
        description="生成直播开场话术模板",
        arguments=[
            {"name": "talent_name", "type": "string", "required": True},
            {"name": "product_name", "type": "string", "required": True},
            {"name": "time_of_day", "type": "string", "required": False},
        ],
        template="""
作为{talent_name}，为直播开场生成一段话术。

产品：{product_name}
时间：{time_of_day}

要求：
1. 热情问候观众
2. 简短介绍今天的产品
3. 营造期待感
4. 风格符合达人人设

请生成开场话术：
"""
    )

    server.register_prompt(
        name="generate_closing",
        description="生成直播结尾话术模板",
        arguments=[
            {"name": "talent_name", "type": "string", "required": True},
            {"name": "call_to_action", "type": "string", "required": True},
        ],
        template="""
作为{talent_name}，为直播结尾生成一段话术。

号召行动：{call_to_action}

要求：
1. 感谢观众
2. 强调产品价值
3. 明确号召行动（{call_to_action}）
4. 营造紧迫感
5. 风格符合达人人设

请生成结尾话术：
"""
    )

    logger.info("ScriptAgent MCP Server fully configured")
    return server
