"""
话术生成Agent - LoRA推理 + 流式输出 + Prompt工程

流程:
  1. 动态加载LoRA Adapter (按垂类切换)
  2. Prompt工程: 注入画像风格、受众特征、场景约束
  3. 流式生成 + 实时敏感词检测
"""

import uuid
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from script_agent.agents.base import BaseAgent
from script_agent.models.message import AgentMessage, GeneratedScript
from script_agent.models.context import (
    CompressedContext,
    InfluencerProfile,
    ProductProfile,
    SessionContext,
)
from script_agent.context.session_compressor import SessionContextCompressor
from script_agent.services.llm_client import (
    LLMServiceClient,
    clean_llm_response,
    GENERATION_DELIMITER,
)
from script_agent.config.settings import settings

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Prompt工程 - 组装生成Prompt
    固定前缀 (Prefix Caching友好) + 动态后缀
    """

    # 场景模板
    SCENARIO_TEMPLATES: Dict[str, str] = {
        "开场话术": """请生成一段直播开场话术，要求：
- 热情打招呼，快速拉近和观众的距离
- 预告今天的直播内容/福利
- 引导观众互动（点赞、关注）
- 时长控制在30秒左右""",

        "产品介绍": """请生成一段产品介绍话术，要求：
- 突出产品核心卖点（前3个）
- 描述使用场景和效果
- 与竞品的差异化对比
- 适当营造紧迫感""",

        "卖点介绍": """请生成一段产品卖点介绍话术，要求：
- 突出产品核心卖点（前3个）
- 描述使用场景和效果
- 与竞品的差异化对比
- 适当营造紧迫感""",

        "促销话术": """请生成一段促销话术，要求：
- 清晰传达优惠信息（价格/折扣/赠品）
- 倒计时/限量制造紧迫感
- 引导立即下单
- 呼应直播间互动""",

        "种草文案": """请生成一段种草文案，要求：
- 个人真实体验视角
- 详细描述产品使用感受
- 适当对比使用前后变化
- 自然种草，避免硬广感""",
    }

    def build(
        self,
        intent_slots: Dict[str, Any],
        profile: InfluencerProfile,
        session: Optional[SessionContext] = None,
        product: Optional[ProductProfile] = None,
        memory_hits: Optional[List[Dict[str, Any]]] = None,
        compressed_context: Optional[CompressedContext] = None,
    ) -> str:
        """构建完整的生成Prompt"""
        parts = []

        # 1. 角色设定 (固定前缀, Prefix Caching可复用)
        parts.append(self._build_role_prompt(profile))

        # 2. 达人风格约束
        parts.append(self._build_style_prompt(profile))

        # 3. 场景任务描述
        scenario = intent_slots.get("sub_scenario") or intent_slots.get("scenario", "")
        parts.append(self._build_scenario_prompt(scenario, intent_slots))

        # 4. 活动信息 (如有)
        event = intent_slots.get("event")
        if event:
            parts.append(f"\n【活动信息】当前活动: {event}，请在话术中融入活动元素。")

        # 4.1 商品信息与卖点约束
        if product:
            parts.append(self._build_product_prompt(product))

        # 4.2 长期记忆向量召回
        if memory_hits:
            parts.append(self._build_memory_prompt(memory_hits))

        # 5. 历史对话摘要 (多轮优化场景)
        if compressed_context and compressed_context.messages:
            parts.append(self._build_compressed_history(compressed_context))
        elif session and session.turns:
            parts.append(self._build_history_context(session))

        # 6. 用户额外要求
        requirements = intent_slots.get("requirements", "")
        if requirements:
            parts.append(f"\n【额外要求】{requirements}")

        # 7. 生成分隔符 — 告诉模型从此处开始输出话术正文
        parts.append(f"\n{GENERATION_DELIMITER}")

        return "\n".join(parts)

    def _build_role_prompt(self, profile: InfluencerProfile) -> str:
        return f"""你是一个专业的电商{profile.category or ''}话术创作专家。
请根据以下达人信息和场景要求，生成高质量的话术内容。
确保内容符合达人个人风格，具有吸引力和互动性，同时遵守平台规范。

重要输出规则：
- 在"{GENERATION_DELIMITER}"标记之后直接输出话术正文
- 不要重复任何提示语、角色设定、风格描述或商品信息
- 不要输出思考过程、分析或格式化标题
- 只输出可以直接在直播/短视频中使用的话术文案"""

    def _build_style_prompt(self, profile: InfluencerProfile) -> str:
        style = profile.style
        lines = ["\n【达人风格】"]
        lines.append(f"- 达人: {profile.name}")
        lines.append(f"- 语气风格: {style.tone or '通用'}")
        if style.catchphrases:
            lines.append(f"- 常用口头禅: {', '.join(style.catchphrases[:5])}")
        lines.append(f"- 正式度: {'口语化' if style.formality_level < 0.4 else '适中' if style.formality_level < 0.7 else '正式'}")
        if style.humor_level > 0.5:
            lines.append("- 适当加入幽默元素")
        if profile.audience_age_range:
            lines.append(f"- 目标受众: {profile.audience_age_range}")
        return "\n".join(lines)

    def _build_scenario_prompt(self, scenario: str,
                                slots: Dict[str, Any]) -> str:
        template = self.SCENARIO_TEMPLATES.get(
            scenario, f"\n请生成一段{scenario or '通用'}话术。"
        )
        category = slots.get("category", "")
        if category:
            template = f"\n【品类】{category}\n" + template
        return template

    def _build_product_prompt(self, product: ProductProfile) -> str:
        lines = ["\n【商品信息】"]
        lines.append(f"- 商品名: {product.name}")
        if product.brand:
            lines.append(f"- 品牌: {product.brand}")
        if product.price_range:
            lines.append(f"- 价格带: {product.price_range}")
        if product.features:
            lines.append(f"- 核心特征: {', '.join(product.features[:6])}")
        if product.selling_points:
            lines.append(f"- 主卖点: {', '.join(product.selling_points[:6])}")
        if product.compliance_notes:
            lines.append(f"- 合规提醒: {'; '.join(product.compliance_notes[:3])}")
        return "\n".join(lines)

    def _build_memory_prompt(self, memory_hits: List[Dict[str, Any]]) -> str:
        # 依据长期记忆预算占比动态调节注入强度，避免记忆片段挤占主任务空间
        total_budget = max(1, int(settings.context.total_token_budget))
        longterm_budget = max(0, int(settings.context.longterm_token_budget))
        budget_ratio = max(0.05, min(0.6, longterm_budget / total_budget))

        if budget_ratio < 0.16:
            max_items = 1
        elif budget_ratio < 0.3:
            max_items = 2
        else:
            max_items = 3
        snippet_max_chars = max(80, min(260, int(110 + 260 * budget_ratio)))

        lines = ["\n【历史高相关样本（向量召回）】"]
        for idx, row in enumerate(memory_hits[:max_items], start=1):
            text = str(row.get("text", "")).replace("\n", " ").strip()
            score = float(row.get("score", 0.0))
            influence = "高参考" if score >= 0.75 else "中参考" if score >= 0.45 else "低参考"
            lines.append(
                f"{idx}. ({influence}, score={score:.3f}) {text[:snippet_max_chars]}"
            )
        lines.append("优先参考高分样本的结构与语气，不要逐句照搬。")
        return "\n".join(lines)

    def _build_history_context(self, session: SessionContext) -> str:
        """构建历史对话摘要 (最近2轮)"""
        recent = session.turns[-2:]
        if not recent:
            return ""
        lines = ["\n【对话上下文】"]
        for t in recent:
            lines.append(f"用户: {t.user_message[:100]}")
            if t.generated_script:
                lines.append(f"上一版话术摘要: {t.generated_script[:100]}...")
        return "\n".join(lines)

    def _build_compressed_history(self, context: CompressedContext) -> str:
        lines = ["\n【压缩会话记忆】"]
        for msg in context.messages[-12:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content[:180]}")
        return "\n".join(lines)


class SensitiveWordFilter:
    """实时敏感词检测 — 复用 quality_agent 的 AC 自动机"""

    def __init__(self):
        from script_agent.agents.quality_agent import SensitiveWordChecker
        self._checker = SensitiveWordChecker()

    def contains_sensitive(self, text: str) -> List[str]:
        passed, issues = self._checker.check(text)
        return [i["word"] for i in issues]


class ScriptGenerationAgent(BaseAgent):
    """
    话术生成Agent
    核心流程: LoRA动态加载 → Prompt工程 → 流式生成 → 敏感词检测
    """

    def __init__(self):
        super().__init__(name="script_generation")
        self.llm = LLMServiceClient()
        self.prompt_builder = PromptBuilder()
        self.sensitive_filter = SensitiveWordFilter()
        self.context_compressor = SessionContextCompressor()
        self.min_chars = max(1, settings.llm.script_min_chars)
        self.primary_attempts = max(1, settings.llm.script_primary_attempts)

    async def process(self, message: AgentMessage) -> AgentMessage:
        intent_slots = message.payload.get("slots", {})
        profile: InfluencerProfile = message.payload.get(
            "profile", InfluencerProfile()
        )
        product: ProductProfile = message.payload.get("product", ProductProfile())
        memory_hits: List[Dict[str, Any]] = message.payload.get("memory_hits", [])
        session: SessionContext = message.payload.get("session", SessionContext())
        category = intent_slots.get("category", profile.category or "通用")

        compressed_context = await self._compress_session_memory(session)
        # 1. 构建Prompt
        prompt = self.prompt_builder.build(
            intent_slots,
            profile,
            session,
            product=product,
            memory_hits=memory_hits,
            compressed_context=compressed_context,
        )
        logger.info(f"Built prompt ({len(prompt)} chars) for category={category}")

        # 2. 调用LLM生成 (失败重试2次后切兜底模型)
        generated_text = await self._generate_with_retry(
            prompt=prompt,
            category=category,
            request_id=message.trace_id or message.message_id,
        )

        # 3. 敏感词检测
        sensitive = self.sensitive_filter.contains_sensitive(generated_text)
        if sensitive:
            logger.warning(f"Sensitive words detected: {sensitive}")
            # 简单替换
            for word in sensitive:
                generated_text = generated_text.replace(word, "***")
        if len(generated_text.strip()) < self.min_chars:
            raise RuntimeError(
                f"generated content too short after filtering: "
                f"{len(generated_text.strip())} < {self.min_chars}"
            )

        # 4. 封装结果
        script = GeneratedScript(
            content=generated_text,
            category=category,
            scenario=intent_slots.get("scenario", ""),
            style_keywords=intent_slots.get("style_hint", "").split(","),
            turn_index=len(session.turns),
            generation_params={
                "prompt_length": len(prompt),
                "memory_hits": len(memory_hits),
                "compressed_context_tokens": (
                    compressed_context.token_count if compressed_context else 0
                ),
            },
        )

        return message.create_response(
            payload={"script": script, "prompt_used": prompt},
            source=self.name,
        )

    async def generate_stream(
        self,
        intent_slots: Dict[str, Any],
        profile: InfluencerProfile,
        session: SessionContext,
        product: Optional[ProductProfile] = None,
        memory_hits: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """流式生成接口 (供API层使用)"""
        category = intent_slots.get("category", profile.category or "通用")
        compressed_context = await self._compress_session_memory(session)
        prompt = self.prompt_builder.build(
            intent_slots,
            profile,
            session,
            product=product,
            memory_hits=memory_hits or [],
            compressed_context=compressed_context,
        )

        content = await self._generate_with_retry(
            prompt=prompt,
            category=category,
            request_id=f"{session.session_id}:stream",
            stream_mode=True,
        )
        if not content or not content.strip():
            logger.warning("generate_stream received empty content from LLM")
            return
        sensitive = self.sensitive_filter.contains_sensitive(content)
        if sensitive:
            logger.warning("Sensitive content detected during streaming")
            yield "[内容涉及敏感词，已终止生成]"
            return
        # 以固定窗口输出，保留流式接口兼容性。
        chunk_size = 16
        for i in range(0, len(content), chunk_size):
            yield content[i:i + chunk_size]

    async def _generate_once(
        self,
        prompt: str,
        category: str,
        request_id: str,
        *,
        prefer_fallback: bool = False,
        stream_mode: bool = False,
    ) -> str:
        if not stream_mode:
            raw = await self.llm.generate_sync(
                prompt,
                category=category,
                max_tokens=800,
                request_id=request_id,
                prefer_fallback=prefer_fallback,
            )
            return clean_llm_response(raw)

        buffer = []
        async for token in self.llm.generate(
            prompt,
            category=category,
            request_id=request_id,
            prefer_fallback=prefer_fallback,
        ):
            buffer.append(token)
        return clean_llm_response("".join(buffer))

    async def _generate_with_retry(
        self,
        prompt: str,
        category: str,
        request_id: str,
        *,
        stream_mode: bool = False,
    ) -> str:
        last_error: Optional[Exception] = None
        for idx in range(1, self.primary_attempts + 1):
            try:
                text = (
                    await self._generate_once(
                        prompt,
                        category,
                        f"{request_id}:primary:{idx}",
                        prefer_fallback=False,
                        stream_mode=stream_mode,
                    )
                ).strip()
                if len(text) < self.min_chars:
                    raise RuntimeError(
                        f"generated content too short: {len(text)} < {self.min_chars}"
                    )
                return text
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "script generation primary attempt failed (%s/%s): %s",
                    idx,
                    self.primary_attempts,
                    exc,
                )

        try:
            fallback_text = (
                await self._generate_once(
                    prompt,
                    category,
                    f"{request_id}:fallback",
                    prefer_fallback=True,
                    stream_mode=stream_mode,
                )
            ).strip()
            if len(fallback_text) < self.min_chars:
                raise RuntimeError(
                    f"fallback generated content too short: "
                    f"{len(fallback_text)} < {self.min_chars}"
                )
            return fallback_text
        except Exception as exc:
            if last_error is None:
                last_error = exc
            raise RuntimeError(
                "script generation failed after retries and fallback"
            ) from last_error

    async def _compress_session_memory(
        self,
        session: SessionContext,
    ) -> Optional[CompressedContext]:
        if not session.turns:
            return None
        # 仅在轮次较多时启用分级压缩，减少额外开销
        if len(session.turns) <= (self.context_compressor.cfg.zone_a_turns + 1):
            return None
        try:
            return await self.context_compressor.compress(session)
        except Exception as exc:
            logger.warning("session memory compression failed: %s", exc)
            return None
