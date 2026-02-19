"""
话术生成Agent - LoRA推理 + 流式输出 + Prompt工程

流程:
  1. 动态加载LoRA Adapter (按垂类切换)
  2. Prompt工程: 注入画像风格、受众特征、场景约束
  3. 流式生成 + 实时敏感词检测
"""

import uuid
import logging
import re
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

        # 3.1 会话任务摘要（提升多轮连贯性）
        if session:
            parts.append(self._build_context_summary(intent_slots, session, product))

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

        # 5.1 续写约束（防止多轮重复）
        if session and session.turns:
            constraints = self._build_continuation_constraints(intent_slots, session)
            if constraints:
                parts.append(constraints)

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

    def _build_context_summary(
        self,
        intent_slots: Dict[str, Any],
        session: SessionContext,
        product: Optional[ProductProfile],
    ) -> str:
        if not session.turns:
            return ""
        lines = ["\n【会话目标摘要】"]
        scenario = (
            intent_slots.get("sub_scenario")
            or intent_slots.get("scenario")
            or "话术生成"
        )
        lines.append(f"- 当前任务: {scenario}")
        if product and product.name:
            lines.append(f"- 当前商品: {product.name}")
        if intent_slots.get("_product_switch"):
            current_name = (
                str(intent_slots.get("product_name", "")).strip()
                or (product.name if product else "")
            )
            previous_name = str(intent_slots.get("_previous_product_name", "")).strip()
            if current_name:
                lines.append(f"- 商品切换: 本轮只围绕“{current_name}”展开")
            if previous_name:
                lines.append(f"- 旧商品禁用: 禁止再提及“{previous_name}”")

        recent_users = [t.user_message.strip() for t in session.turns[-3:] if t.user_message.strip()]
        if recent_users:
            lines.append("- 最近用户诉求:")
            for q in recent_users[-2:]:
                lines.append(f"  - {q[:48]}")

        last_script = ""
        for turn in reversed(session.turns):
            if turn.generated_script and turn.generated_script.strip():
                last_script = turn.generated_script.strip()
                break
        if last_script:
            lines.append(f"- 上版话术核心: {last_script[:90]}")
        lines.append("- 本轮要求: 语气保持一致，但信息表达要有新增，不要整段复述")
        return "\n".join(lines)

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
        """构建历史对话摘要 (最近3轮)"""
        recent = session.turns[-3:]
        if not recent:
            return ""
        lines = ["\n【对话上下文】"]
        for t in recent:
            lines.append(f"用户: {t.user_message[:100]}")
            if t.generated_script:
                lines.append(f"上一版话术摘要: {t.generated_script[:100]}...")
        return "\n".join(lines)

    def _build_continuation_constraints(
        self,
        slots: Dict[str, Any],
        session: SessionContext,
    ) -> str:
        requirements = str(slots.get("requirements", ""))
        continuation = bool(slots.get("_continuation")) or any(
            kw in requirements for kw in ("继续", "再来", "补充", "续写", "换一个")
        )
        if not continuation:
            return (
                "\n【重复抑制】\n"
                "- 同一句话不要重复出现\n"
                "- 禁止输出提示词片段、字段清单、占位符（如 --- / ...）"
            )
        prev_head = ""
        for t in reversed(session.turns):
            if t.generated_script and t.generated_script.strip():
                prev_head = t.generated_script.strip()[:24]
                break
        lines = ["\n【续写约束】"]
        lines.append("- 当前为续写任务：保持同一达人风格，但必须输出新增信息")
        if prev_head:
            lines.append(f"- 不要复用上版开头（例如“{prev_head}”）")
        lines.append("- 新增至少2个信息点（卖点/场景/互动/转化引导）")
        lines.append("- 不输出字段清单、提示语、占位符（如 --- / ...）")
        if slots.get("_product_switch"):
            new_product = str(slots.get("product_name", "")).strip()
            old_product = str(slots.get("_previous_product_name", "")).strip()
            if new_product:
                lines.append(f"- 必须明确提及新商品“{new_product}”")
            if old_product:
                lines.append(f"- 绝对不要出现旧商品“{old_product}”")
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
        # 线上模型波动时，至少重试 3 次可明显降低空结果/截断概率。
        self.primary_attempts = max(3, settings.llm.script_primary_attempts)
        self._separator_only_re = re.compile(r"^[-—_=*`.\s]{2,}$")
        self._meta_line_re = re.compile(
            r"^\s*-\s*(?:\*\*)?"
            r"(达人|语气风格|常用口头禅|正式度|目标受众|商品名|品牌|价格带|核心特征|主卖点|合规提醒|核心卖点)"
            r"(?:\*\*)?\s*[:：]"
        )
        self._meta_heading_re = re.compile(
            r"^\s*#{1,6}\s*(商品名|内容描述|核心卖点|商品信息|达人风格|会话目标摘要|对话上下文|重复抑制|续写约束|活动信息|额外要求)(?:\s*[:：].*)?\s*$"
        )
        self._meta_sentence_re = re.compile(
            r"^\s*(?:\*\*)?(本轮要求|上一版话术摘要|上版话术核心)(?:\*\*)?\s*[:：]"
        )
        self._inline_prompt_echo_patterns = (
            re.compile(
                r"[（(【\[]?\s*本轮要求\s*[:：]?\s*"
                r"[^。！？!?\n）)】\]]{0,180}[）)】\]]?"
            ),
            re.compile(
                r"[（(【\[]?\s*(?:上一版话术摘要|上版话术核心)\s*[:：]?\s*"
                r"[^。！？!?\n）)】\]]{0,220}[）)】\]]?"
            ),
            re.compile(
                r"[（(【\[]?\s*语气保持一致"
                r"[^。！？!?\n）)】\]]{0,220}?不要整段复述[）)】\]]?"
            ),
        )
        self._tail_incomplete_suffixes = (
            "的", "了", "和", "与", "及", "并", "在", "把", "将", "并且", "以及", "等",
        )
        self._tail_soft_complete_endings = (
            "吧", "呀", "啊", "呢", "哦", "啦", "喔", "哈", "~", "～", "…",
        )
        self._tail_hard_incomplete_re = re.compile(r"(比如|例如|包括|例如说|像)$")

    async def process(self, message: AgentMessage) -> AgentMessage:
        intent_slots = dict(message.payload.get("slots", {}))
        profile: InfluencerProfile = message.payload.get(
            "profile", InfluencerProfile()
        )
        product: ProductProfile = message.payload.get("product", ProductProfile())
        memory_hits: List[Dict[str, Any]] = message.payload.get("memory_hits", [])
        session: SessionContext = message.payload.get("session", SessionContext())
        feedback = message.payload.get("feedback", [])
        if feedback:
            intent_slots["requirements"] = self._merge_feedback_requirements(
                str(intent_slots.get("requirements", "")),
                [str(x).strip() for x in feedback if str(x).strip()],
            )
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
            session=session,
            slots=intent_slots,
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
                "product_name": str(intent_slots.get("product_name", "")).strip(),
                "product_switch": bool(intent_slots.get("_product_switch")),
                "previous_product_name": str(
                    intent_slots.get("_previous_product_name", "")
                ).strip(),
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
            session=session,
            slots=intent_slots,
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
        max_tokens: Optional[int] = None,
    ) -> str:
        max_tokens = max_tokens or settings.llm.max_tokens
        if not stream_mode:
            raw = await self.llm.generate_sync(
                prompt,
                category=category,
                max_tokens=max_tokens,
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
            max_tokens=max_tokens,
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
        session: Optional[SessionContext] = None,
        slots: Optional[Dict[str, Any]] = None,
    ) -> str:
        last_error: Optional[Exception] = None
        last_reason = ""
        last_text = ""
        best_soft_candidate = ""
        base_max_tokens = self._resolve_max_tokens(category, stream_mode=stream_mode)
        for idx in range(1, self.primary_attempts + 1):
            try:
                attempt_prompt = self._build_retry_prompt(
                    base_prompt=prompt,
                    attempt=idx,
                    reason=last_reason,
                    previous_text=last_text,
                )
                attempt_max_tokens = self._resolve_attempt_max_tokens(
                    base=base_max_tokens,
                    attempt=idx,
                    reason=last_reason,
                    stream_mode=stream_mode,
                )
                text = (
                    await self._generate_once(
                        attempt_prompt,
                        category,
                        f"{request_id}:primary:{idx}",
                        prefer_fallback=False,
                        stream_mode=stream_mode,
                        max_tokens=attempt_max_tokens,
                    )
                ).strip()
                text = self._post_process_output(text)
                text = self._reduce_leading_overlap(text, session, slots or {})
                issue = self._validate_generation_output(
                    text=text,
                    session=session,
                    slots=slots or {},
                )
                if issue:
                    if self._is_soft_issue(issue) and len(text) >= self.min_chars:
                        best_soft_candidate = text
                    last_reason = issue
                    last_text = text
                    raise RuntimeError(issue)
                if len(text) < self.min_chars:
                    last_reason = "content_too_short"
                    last_text = text
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
                    self._build_retry_prompt(
                        base_prompt=prompt,
                        attempt=self.primary_attempts + 1,
                        reason=last_reason or "primary_failed",
                        previous_text=last_text,
                    ),
                    category,
                    f"{request_id}:fallback",
                    prefer_fallback=True,
                    stream_mode=stream_mode,
                    max_tokens=self._resolve_attempt_max_tokens(
                        base=base_max_tokens,
                        attempt=self.primary_attempts + 1,
                        reason=last_reason or "primary_failed",
                        stream_mode=stream_mode,
                    ),
                )
            ).strip()
            fallback_text = self._post_process_output(fallback_text)
            fallback_text = self._reduce_leading_overlap(
                fallback_text, session, slots or {}
            )
            fallback_issue = self._validate_generation_output(
                text=fallback_text,
                session=session,
                slots=slots or {},
            )
            if fallback_issue:
                raise RuntimeError(f"fallback content invalid: {fallback_issue}")
            if len(fallback_text) < self.min_chars:
                raise RuntimeError(
                    f"fallback generated content too short: "
                    f"{len(fallback_text)} < {self.min_chars}"
                )
            return fallback_text
        except Exception as exc:
            if last_error is None:
                last_error = exc
            if best_soft_candidate:
                logger.warning(
                    "script generation fallback failed, return best soft candidate: reason=%s",
                    last_reason or "unknown",
                )
                return best_soft_candidate
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

    def _build_retry_prompt(
        self,
        *,
        base_prompt: str,
        attempt: int,
        reason: str,
        previous_text: str,
    ) -> str:
        if attempt <= 1:
            return base_prompt
        lines = [base_prompt, "\n【重试修正要求】"]
        if reason == "prompt_echo_detected":
            lines.append("- 禁止输出字段清单或配置项（如“商品名/主卖点/合规提醒”）")
        elif reason == "high_overlap_with_recent_script":
            lines.append("- 与上一版重复过高，请改写为全新表述，不要复用开头")
        elif reason == "missing_new_product_name":
            lines.append("- 你没有提及本轮新商品名，请明确写出新商品并围绕其卖点展开")
        elif reason == "contains_previous_product_name":
            lines.append("- 输出中出现了旧商品名，请删除旧商品并仅保留新商品信息")
        elif reason == "effective_content_too_short":
            lines.append("- 正文信息不足，请扩展为完整口播段，至少包含卖点、场景和互动引导")
        elif reason == "tail_incomplete":
            lines.append("- 你的结尾不完整，请补全最后一句并以完整句号或感叹号收尾")
        elif reason == "content_too_short":
            lines.append("- 内容长度不足，请补足完整口播段落并加入互动与转化信息")
        elif reason == "repetition_detected":
            lines.append("- 句子重复明显，请删除重复句并补充新信息点")
        else:
            lines.append("- 请直接输出可用话术正文，不要输出提示语或元信息")
        if previous_text:
            lines.append(f"- 上次无效输出片段（禁止复用）：{previous_text[:70]}")
        return "\n".join(lines)

    def _post_process_output(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        lines = []
        for line in text.splitlines():
            s = line.strip()
            if self._separator_only_re.match(s):
                continue
            if s in {"---", "...", "…"}:
                continue
            if self._meta_line_re.match(s):
                continue
            if self._meta_heading_re.match(s):
                continue
            if self._meta_sentence_re.match(s):
                continue
            lines.append(line)
        cleaned = "\n".join(lines).strip()
        cleaned = self._strip_inline_prompt_echo(cleaned)
        return self._dedupe_sentences(cleaned)

    def _dedupe_sentences(self, text: str) -> str:
        parts = re.findall(r"[^。！？!?；;\n]+[。！？!?；;\n]?", text)
        if not parts:
            return text
        seen = set()
        result = []
        for part in parts:
            normalized = self._normalize_text(part)
            if len(normalized) >= 6 and normalized in seen:
                continue
            if len(normalized) >= 6:
                seen.add(normalized)
            result.append(part)
        return "".join(result).strip()

    def _validate_generation_output(
        self,
        *,
        text: str,
        session: Optional[SessionContext],
        slots: Dict[str, Any],
    ) -> Optional[str]:
        if not text:
            return "content_too_short"
        if self._is_prompt_echo(text):
            return "prompt_echo_detected"
        if self._effective_content_length(text) < self.min_chars:
            return "effective_content_too_short"
        if self._is_tail_incomplete(text):
            return "tail_incomplete"
        if self._has_heavy_repetition(text):
            return "repetition_detected"
        product_switch_issue = self._check_product_switch_consistency(text, slots)
        if product_switch_issue:
            return product_switch_issue
        if self._is_high_overlap_with_recent(text, session, slots):
            return "high_overlap_with_recent_script"
        return None

    def _is_prompt_echo(self, text: str) -> bool:
        inline_markers = (
            "本轮要求",
            "上一版话术摘要",
            "上版话术核心",
            "信息表达要有新增",
            "不要整段复述",
        )
        if any(marker in text for marker in inline_markers):
            return True
        marker_hits = 0
        markers = ("商品名", "主卖点", "核心特征", "合规提醒", "语气风格", "达人")
        for m in markers:
            marker_hits += text.count(m)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        bullet_like = sum(1 for ln in lines[:4] if ln.startswith("-"))
        markdown_heading = sum(1 for ln in lines[:4] if ln.startswith("#"))
        meta_like = sum(
            1 for ln in lines[:6]
            if self._meta_line_re.match(ln)
            or self._meta_heading_re.match(ln)
            or self._meta_sentence_re.match(ln)
        )
        return (
            (marker_hits >= 3 and bullet_like >= 2)
            or (meta_like >= 2 and marker_hits >= 1)
            or (markdown_heading >= 1 and marker_hits >= 1)
        )

    def _strip_inline_prompt_echo(self, text: str) -> str:
        cleaned = text or ""
        for pattern in self._inline_prompt_echo_patterns:
            cleaned = pattern.sub("", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _effective_content_length(self, text: str) -> int:
        if not text:
            return 0
        cleaned = re.sub(
            r"(?:\*\*)?(本轮要求|上一版话术摘要|上版话术核心)(?:\*\*)?\s*[:：][^\n。！？!?]*",
            "",
            text,
        )
        cleaned = re.sub(
            r"(?:---话术正文---|###\s*(商品名|内容描述|核心卖点)[^。\n]*|-\s*\*\*(核心卖点|主卖点)\*\*[:：][^\n]*)",
            "",
            cleaned,
        )
        normalized = self._normalize_text(cleaned)
        return len(normalized)

    def _is_tail_incomplete(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return True
        if stripped.endswith(("。", "！", "？", "!", "?", "；", ";", "”", "\"", "）", ")")):
            return False
        if stripped.endswith(self._tail_soft_complete_endings):
            return False
        parts = re.split(r"[。！？!?；;\n]", stripped)
        tail = parts[-1].strip() if parts else stripped
        normalized_tail = self._normalize_text(tail)
        if not normalized_tail:
            return True
        if len(normalized_tail) < 5:
            return True
        # 放宽判定边界: 长正文 + 非强截断尾巴，视为可接受。
        normalized_full = self._normalize_text(stripped)
        if len(normalized_full) >= max(self.min_chars * 2, 80):
            if len(normalized_tail) >= 4 and not self._tail_hard_incomplete_re.search(normalized_tail):
                return False
        # 已有多句完整输出时，最后一句轻微口语收尾不强制判失败。
        sentence_count = len(
            [p for p in parts if self._normalize_text(p.strip())]
        )
        if sentence_count >= 2 and len(normalized_tail) >= 4:
            if not any(normalized_tail.endswith(sfx) for sfx in self._tail_incomplete_suffixes):
                return False
        if any(normalized_tail.endswith(sfx) for sfx in self._tail_incomplete_suffixes):
            return True
        if len(normalized_tail) >= 14:
            return False
        return False

    def _check_product_switch_consistency(
        self,
        text: str,
        slots: Dict[str, Any],
    ) -> Optional[str]:
        if not slots.get("_product_switch"):
            return None
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            return "content_too_short"
        new_product = self._normalize_text(str(slots.get("product_name", "")).strip())
        old_product = self._normalize_text(
            str(slots.get("_previous_product_name", "")).strip()
        )
        if new_product and new_product not in normalized_text:
            return "missing_new_product_name"
        if old_product and old_product in normalized_text and old_product != new_product:
            return "contains_previous_product_name"
        return None

    def _has_heavy_repetition(self, text: str) -> bool:
        parts = [p for p in re.findall(r"[^。！？!?；;\n]+", text) if p.strip()]
        if len(parts) < 3:
            return False
        seen = {}
        for part in parts:
            key = self._normalize_text(part)
            if len(key) < 8:
                continue
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 2:
                return True
        return False

    def _is_high_overlap_with_recent(
        self,
        text: str,
        session: Optional[SessionContext],
        slots: Dict[str, Any],
    ) -> bool:
        if not session or not session.turns:
            return False
        recent_scripts = []
        for turn in reversed(session.turns):
            if turn.generated_script and turn.generated_script.strip():
                recent_scripts.append(turn.generated_script.strip())
            if len(recent_scripts) >= 2:
                break
        if not recent_scripts:
            return False
        requirements = str(slots.get("requirements", ""))
        is_continuation = bool(slots.get("_continuation")) or any(
            kw in requirements for kw in ("继续", "再来", "补充", "续写", "换一个")
        )
        if is_continuation:
            current_prefix = self._leading_sentence(text)
            if current_prefix:
                norm_current = self._normalize_text(current_prefix)
                for prev in recent_scripts:
                    norm_prev = self._normalize_text(self._leading_sentence(prev))
                    if norm_current and norm_prev and norm_current == norm_prev:
                        return True
        threshold = 0.68 if is_continuation else 0.82
        return any(
            self._char_ngram_jaccard(text, prev) >= threshold
            for prev in recent_scripts
        )

    def _char_ngram_jaccard(self, a: str, b: str, n: int = 4) -> float:
        na = self._char_ngrams(self._normalize_text(a), n)
        nb = self._char_ngrams(self._normalize_text(b), n)
        if not na or not nb:
            return 0.0
        return len(na & nb) / len(na | nb)

    def _char_ngrams(self, text: str, n: int) -> set:
        if not text:
            return set()
        if len(text) <= n:
            return {text}
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "").lower()

    def _is_soft_issue(self, issue: str) -> bool:
        return issue in {"high_overlap_with_recent_script", "repetition_detected"}

    def _merge_feedback_requirements(
        self,
        current_requirements: str,
        feedback: List[str],
    ) -> str:
        current = (current_requirements or "").strip()
        if not feedback:
            return current
        tips = "；".join(dict.fromkeys(feedback))
        if current:
            return f"{current}；质量修正要求：{tips}"[:2000]
        return f"质量修正要求：{tips}"[:2000]

    def _resolve_max_tokens(self, category: str, *, stream_mode: bool) -> int:
        backend_name = type(self.llm.backend).__name__.lower()
        is_zhipu = "zhipu" in backend_name or settings.llm.primary_backend.lower() == "zhipu"
        if stream_mode:
            base = max(settings.llm.max_tokens, 1400)
            if is_zhipu:
                base = max(base, 2200)
            return min(3200, base)
        base = max(1000, min(settings.llm.max_tokens, 1800))
        if is_zhipu:
            base = max(base, 1800)
        return min(2800, base)

    def _resolve_attempt_max_tokens(
        self,
        *,
        base: int,
        attempt: int,
        reason: str,
        stream_mode: bool,
    ) -> int:
        max_limit = 3200 if stream_mode else 2800
        boost_reasons = {
            "prompt_echo_detected",
            "effective_content_too_short",
            "content_too_short",
            "tail_incomplete",
        }
        if reason in boost_reasons:
            base = int(base * 1.35)
        if attempt >= 2:
            base = int(base * 1.15)
        if reason == "tail_incomplete":
            base += 240 if stream_mode else 180
        if attempt >= 3:
            base += 120
        return min(max_limit, max(384, base))

    def _reduce_leading_overlap(
        self,
        text: str,
        session: Optional[SessionContext],
        slots: Dict[str, Any],
    ) -> str:
        if not text or not session or not session.turns:
            return text
        requirements = str(slots.get("requirements", ""))
        is_continuation = bool(slots.get("_continuation")) or any(
            kw in requirements for kw in ("继续", "再来", "补充", "续写", "换一个")
        )
        if not is_continuation:
            return text
        prev_script = ""
        for turn in reversed(session.turns):
            if turn.generated_script and turn.generated_script.strip():
                prev_script = turn.generated_script.strip()
                break
        if not prev_script:
            return text
        current_head = self._leading_sentence(text)
        prev_head = self._leading_sentence(prev_script)
        if not current_head or not prev_head:
            return text
        if self._normalize_text(current_head) != self._normalize_text(prev_head):
            return text
        parts = re.findall(r"[^。！？!?；;\n]+[。！？!?；;\n]?", text)
        if len(parts) <= 1:
            return text
        trimmed = "".join(parts[1:]).strip()
        if len(trimmed) >= self.min_chars:
            return trimmed
        return text

    def _leading_sentence(self, text: str) -> str:
        parts = re.findall(r"[^。！？!?；;\n]+[。！？!?；;\n]?", text or "")
        if not parts:
            return ""
        return parts[0].strip()
