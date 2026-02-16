"""
系统层上下文管理器

设计要点:
  1. 模板化 — 系统Prompt由多个模块拼装, 按需组合
  2. 分层缓存 — 模板本地缓存, 配置中心远程更新
  3. Prefix Caching友好 — 固定前缀部分尽量不变
"""

import logging
from typing import Dict

from script_agent.models.context import SystemContext

logger = logging.getLogger(__name__)


class SystemContextManager:
    """
    系统层上下文管理器
    构建: 基础角色 + 垂类知识 + 合规约束 + 输出格式 + 租户配置
    """

    def __init__(self):
        self.template_cache: Dict[str, str] = {}

    async def build(self, tenant_id: str = "", category: str = "",
                    scenario: str = "") -> SystemContext:
        """构建系统层上下文"""
        base_role = self._get_base_role()
        category_knowledge = self._get_category_knowledge(category)
        compliance_rules = self._get_compliance_rules(category)
        output_format = self._get_output_format(scenario)
        tenant_config = await self._get_tenant_config(tenant_id)

        prompt = self._compose_system_prompt(
            base_role, category_knowledge, compliance_rules,
            output_format, tenant_config,
        )

        return SystemContext(
            base_role=base_role,
            category_knowledge=category_knowledge,
            compliance_rules=compliance_rules,
            output_format=output_format,
            tenant_config=tenant_config,
            prompt=prompt,
        )

    # ------------------------------------------------------------------

    def _get_base_role(self) -> str:
        """基础角色定义 — 所有请求共享, 不变 (Prefix Caching命中率最高)"""
        return (
            "你是一个专业的电商达人话术创作助手。\n"
            "你的职责是根据达人的风格特点和业务场景，生成高质量的直播话术、短视频脚本等内容。\n"
            "你需要确保生成的内容：\n"
            "1. 符合达人的个人风格和语言习惯\n"
            "2. 匹配指定的业务场景（直播开场/促销/种草等）\n"
            "3. 遵守平台规范和广告法要求\n"
            "4. 具有吸引力和互动性"
        )

    def _get_category_knowledge(self, category: str) -> str:
        """垂类专业知识 — 按品类切换"""
        templates = {
            "美妆": (
                "【美妆品类知识】\n"
                "- 产品描述重点：质地、色号、持妆时间、适合肤质\n"
                "- 话术风格倾向：感性描述为主，强调使用体验和效果对比\n"
                "- 常用表达：上脸效果、显白、不卡粉、一整天不脱妆\n"
                "- 禁止表达：医疗功效声称、绝对化用语"
            ),
            "食品": (
                "【食品品类知识】\n"
                "- 产品描述重点：口感、原料、保质期、食用场景\n"
                "- 话术风格倾向：生活化、场景化，强调味觉体验\n"
                "- 常用表达：入口即化、回味无穷、家庭必备\n"
                "- 禁止表达：药效声称、夸大营养功能"
            ),
            "服饰": (
                "【服饰品类知识】\n"
                "- 产品描述重点：面料、版型、适合场景、搭配建议\n"
                "- 话术风格倾向：视觉化描述，强调穿搭效果\n"
                "- 常用表达：显瘦、百搭、质感拉满、通勤必备\n"
                "- 禁止表达：绝对化承诺"
            ),
            "数码": (
                "【数码品类知识】\n"
                "- 产品描述重点：参数、性能、使用场景、性价比\n"
                "- 话术风格倾向：理性分析+感性体验结合\n"
                "- 常用表达：性能拉满、颜值在线、办公利器\n"
                "- 禁止表达：虚假参数、误导对比"
            ),
        }
        return templates.get(category, "")

    def _get_compliance_rules(self, category: str) -> str:
        """合规约束"""
        base = (
            "【合规要求】\n"
            "- 禁止使用广告法禁止的极限词（最、第一、唯一等）\n"
            "- 禁止虚假宣传和夸大功效\n"
            "- 数据/效果声称需有依据\n"
            "- 不得贬低竞品"
        )
        if category == "美妆":
            base += "\n- 不得声称医疗/治疗效果\n- 成分功效需符合化妆品管理条例"
        elif category == "食品":
            base += "\n- 不得声称保健/药用功能\n- 营养声称需符合GB标准"
        return base

    def _get_output_format(self, scenario: str) -> str:
        """输出格式定义"""
        formats = {
            "直播带货": "话术长度200-400字，分段落输出，标注互动点和节奏提示。",
            "短视频": "文案长度100-200字，节奏紧凑，开头3秒抓住注意力。",
            "种草文案": "图文并茂风格，300-500字，个人体验视角。",
        }
        return formats.get(scenario, "话术长度200-400字，条理清晰。")

    async def _get_tenant_config(self, tenant_id: str) -> str:
        """租户定制配置 (实际从配置中心获取)"""
        # 模拟
        if tenant_id:
            return f"【租户配置】租户ID: {tenant_id}"
        return ""

    def _compose_system_prompt(self, *parts: str) -> str:
        """
        拼装System Prompt
        
        设计: 固定前缀 + 可变后缀
        ┌────────────────────────┐
        │ 固定前缀 (base_role)    │ ← Prefix Caching复用
        ├────────────────────────┤
        │ 可变后缀               │ ← 按品类/场景变化
        └────────────────────────┘
        """
        return "\n\n".join(p for p in parts if p)
