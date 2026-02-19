"""
话术生成 Skill — 将现有核心生成流程包装为 Skill
"""

import re
from typing import Any, Dict

from script_agent.skills.base import BaseSkill, SkillContext, SkillResult
from script_agent.agents.script_agent import ScriptGenerationAgent
from script_agent.agents.quality_agent import QualityCheckAgent
from script_agent.models.message import (
    AgentMessage,
    GeneratedScript,
    MessageType,
    QualityResult,
)
from script_agent.config.settings import settings


class ScriptGenerationSkill(BaseSkill):
    """话术生成 Skill — 核心业务"""

    name = "script_generation"
    display_name = "话术生成"
    description = "根据品类、场景、达人风格生成话术"
    required_slots = ["category", "scenario"]
    input_schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "minLength": 1, "maxLength": 32},
            "scenario": {"type": "string", "minLength": 1, "maxLength": 64},
            "sub_scenario": {"type": "string", "maxLength": 64},
            "event": {"type": "string", "maxLength": 64},
            "requirements": {"type": "string", "maxLength": 2000},
            "style_hint": {"type": "string", "maxLength": 200},
            "product_name": {"type": "string", "maxLength": 128},
            "product_id": {"type": "string", "maxLength": 64},
            "target_name": {"type": "string", "maxLength": 64},
            "brand": {"type": "string", "maxLength": 64},
            "price_range": {"type": "string", "maxLength": 64},
            "target_audience": {"type": "string", "maxLength": 128},
            "selling_points": {
                "type": ["array", "string"],
                "items": {"type": "string", "maxLength": 120},
            },
            "product_features": {
                "type": ["array", "string"],
                "items": {"type": "string", "maxLength": 120},
            },
            "_raw_query": {"type": "string", "maxLength": 2000},
            "_continuation": {"type": "boolean"},
            "_category_source": {"type": "string", "maxLength": 32},
            "_product_switch": {"type": "boolean"},
            "_previous_product_name": {"type": "string", "maxLength": 128},
            "_intent_adjusted": {"type": "string", "maxLength": 64},
            "intent": {"type": "string", "maxLength": 64},
            "target_id": {"type": "string", "maxLength": 64},
        },
        "required": ["category", "scenario"],
        "additionalProperties": False,
    }

    def __init__(self):
        self._script_agent = ScriptGenerationAgent()
        self._quality_agent = QualityCheckAgent()

    def can_handle(self, intent: str, slots: Dict[str, Any]) -> float:
        if intent == "script_generation":
            return 1.0
        if intent in ("script_optimization",):
            return 0.6
        return 0.0

    def _degraded_success(
        self,
        *,
        script: GeneratedScript,
        quality_result,
        message: str,
    ) -> SkillResult:
        return SkillResult(
            success=True,
            script=script,
            quality_result=quality_result,
            message=message,
            data={"degraded": True},
        )

    def _extract_product_name(self, context: SkillContext) -> str:
        slots = context.intent.slots or {}
        product = context.extra.get("product")
        candidates = [
            str(slots.get("product_name", "")).strip(),
            str(getattr(product, "name", "") or "").strip(),
        ]
        for c in candidates:
            if c:
                return c
        q = str(context.query or slots.get("_raw_query", "")).strip()
        if not q:
            return ""
        m = re.search(r"(?:介绍|推荐|讲讲|说说|改成|换成)([^，。；\n]{2,20})", q)
        if m:
            return m.group(1).strip()
        return ""

    def _collect_points(self, context: SkillContext) -> list[str]:
        slots = context.intent.slots or {}
        product = context.extra.get("product")
        points: list[str] = []

        raw_points = slots.get("selling_points") or slots.get("product_features")
        if isinstance(raw_points, str):
            points.extend([p.strip() for p in re.split(r"[、,，;；\s]+", raw_points) if p.strip()])
        elif isinstance(raw_points, list):
            points.extend([str(p).strip() for p in raw_points if str(p).strip()])

        product_features = getattr(product, "features", None) or []
        points.extend([str(p).strip() for p in product_features if str(p).strip()])

        # 去重且保留顺序
        seen = set()
        deduped = []
        for p in points:
            if p in seen:
                continue
            seen.add(p)
            deduped.append(p)
            if len(deduped) >= 3:
                break
        return deduped

    def _build_local_fallback_script(
        self,
        context: SkillContext,
        *,
        reason: str,
    ) -> GeneratedScript:
        slots = context.intent.slots or {}
        profile = context.profile
        session = context.session
        category = str(slots.get("category") or profile.category or "通用").strip() or "通用"
        scenario = str(slots.get("scenario") or "直播带货").strip() or "直播带货"
        product_name = self._extract_product_name(context) or "这款商品"
        points = self._collect_points(context)

        style_hint = str(slots.get("style_hint", "")).lower()
        tone = "各位朋友" if ("professional" in style_hint or "专业" in str(context.query)) else "家人们"
        points_text = "、".join(points[:2]) if points else "口感体验、使用场景和性价比"

        content = (
            f"{tone}，今天重点给大家介绍{product_name}。"
            f"它非常适合{scenario}场景来讲解，核心亮点可以抓住{points_text}。"
            f"下单建议是先锁单再看福利，直播间会按节奏补充细节与优惠，"
            "有需要的朋友可以直接留言，我按你的风格继续优化成可直接口播的版本。"
        )

        # 保证不低于最小字数约束
        if len(content.strip()) < settings.llm.script_min_chars:
            content += "现在下单还有机会领取限时加赠，建议尽快加入购物车避免错过。"

        return GeneratedScript(
            content=content,
            category=category,
            scenario=scenario,
            style_keywords=[str(slots.get("style_hint", "")).strip()] if slots.get("style_hint") else [],
            turn_index=len(session.turns),
            quality_score=0.55,
            generation_params={
                "source": "local_fallback",
                "reason": reason,
                "product_name": product_name,
            },
        )

    async def execute(self, context: SkillContext) -> SkillResult:
        slots = context.intent.slots
        profile = context.profile
        session = context.session
        product = context.extra.get("product")
        memory_hits = context.extra.get("memory_hits", [])

        # 生成
        script_msg = AgentMessage(
            trace_id=context.trace_id,
            payload={
                "slots": slots,
                "profile": profile,
                "product": product,
                "memory_hits": memory_hits,
                "session": session,
            },
            session_id=session.session_id,
        )
        script_resp = await self._script_agent(script_msg)
        if script_resp.message_type == MessageType.ERROR:
            degraded = self._build_local_fallback_script(
                context,
                reason=script_resp.payload.get("error_message", "script_agent_error"),
            )
            return self._degraded_success(
                script=degraded,
                quality_result=QualityResult(
                    passed=False,
                    overall_score=degraded.quality_score,
                    suggestions=["模型异常时已回退本地兜底文案，建议点击重新生成获取更高质量版本"],
                ),
                message=script_resp.payload.get("error_message", "话术生成失败") + "，已启用本地兜底文案",
            )
        script: GeneratedScript = script_resp.payload.get("script", GeneratedScript())
        min_chars = max(1, settings.llm.script_min_chars)
        if not script.content.strip():
            degraded = self._build_local_fallback_script(
                context,
                reason="empty_script",
            )
            return self._degraded_success(
                script=degraded,
                quality_result=QualityResult(
                    passed=False,
                    overall_score=degraded.quality_score,
                    suggestions=["模型返回空结果，已自动回退兜底文案"],
                ),
                message="生成结果为空，已启用本地兜底文案",
            )
        if len(script.content.strip()) < min_chars:
            degraded = self._build_local_fallback_script(
                context,
                reason="short_script",
            )
            return self._degraded_success(
                script=degraded,
                quality_result=QualityResult(
                    passed=False,
                    overall_score=degraded.quality_score,
                    suggestions=[f"模型结果不足{min_chars}字，已自动回退兜底文案"],
                ),
                message=f"生成结果不足{min_chars}字，已启用本地兜底文案",
            )

        # 质量校验 + 重试
        retry_count = 0
        quality_result = None
        while retry_count <= settings.quality.max_retries:
            quality_msg = AgentMessage(
                trace_id=context.trace_id,
                payload={"script": script, "profile": profile},
                session_id=session.session_id,
            )
            quality_resp = await self._quality_agent(quality_msg)
            if quality_resp.message_type == MessageType.ERROR:
                if script.content.strip():
                    return self._degraded_success(
                        script=script,
                        quality_result=quality_result,
                        message=(
                            quality_resp.payload.get("error_message", "质量检查失败")
                            + "，已返回可用文案"
                        ),
                    )
                return SkillResult(
                    success=False,
                    script=script,
                    message=quality_resp.payload.get("error_message", "质量检查失败"),
                )
            quality_result = quality_resp.payload.get("quality_result")

            if quality_result and quality_result.passed:
                break

            retry_count += 1
            if retry_count <= settings.quality.max_retries:
                script_msg.payload["feedback"] = (
                    quality_result.suggestions if quality_result else []
                )
                previous_script = script
                script_resp = await self._script_agent(script_msg)
                if script_resp.message_type == MessageType.ERROR:
                    if script.content.strip():
                        return self._degraded_success(
                            script=script,
                            quality_result=quality_result,
                            message=(
                                script_resp.payload.get("error_message", "话术生成失败")
                                + "，已回退到上一版可用文案"
                            ),
                        )
                    return SkillResult(
                        success=False,
                        script=script,
                        message=script_resp.payload.get("error_message", "话术生成失败"),
                    )
                script = script_resp.payload.get("script", script)
                if len(script.content.strip()) < min_chars:
                    if previous_script and len(previous_script.content.strip()) >= min_chars:
                        return self._degraded_success(
                            script=previous_script,
                            quality_result=quality_result,
                            message=f"重试后结果不足{min_chars}字，已返回上一版可用文案",
                        )
                    degraded = self._build_local_fallback_script(
                        context,
                        reason="retry_short_script",
                    )
                    return self._degraded_success(
                        script=degraded,
                        quality_result=quality_result,
                        message=f"重试后结果不足{min_chars}字，已启用本地兜底文案",
                    )

        if quality_result and not quality_result.passed and script.content.strip():
            return self._degraded_success(
                script=script,
                quality_result=quality_result,
                message="质量校验未完全通过，已返回可用文案",
            )
        return SkillResult(
            success=quality_result.passed if quality_result else False,
            script=script,
            quality_result=quality_result,
        )
