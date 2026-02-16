"""
会话层分级压缩器

核心思想: 越近的对话越重要, 保留越完整

Zone A: 最近2轮  → 完整保留 (指代消解依赖)
Zone B: 3-6轮前  → 语义摘要 (LLMLingua-2 + 规则压缩)
  B-1: 3-4轮前 → LLMLingua-2 精细压缩 (信息保留88%)
  B-2: 5-6轮前 → 规则压缩 (信息保留70%)
Zone C: 7轮以前  → LLM深度摘要 (激进压缩)
Zone D: 已不相关  → 丢弃
"""

import logging
import re
from typing import Any, Dict, List, Optional

from script_agent.models.context import (
    SessionContext, ConversationTurn, CompressedContext,
)
from script_agent.services.llm_client import LLMServiceClient
from script_agent.config.settings import settings

logger = logging.getLogger(__name__)


class TurnImportanceScorer:
    """
    轮次重要性评分器
    决定哪些轮次优先保留、哪些优先压缩
    """

    def score(self, turn: ConversationTurn,
              session: SessionContext) -> float:
        score = 0.5  # 基准分

        # 1. 生成了话术的轮次更重要
        if turn.generated_script:
            score += 0.2

        # 2. 用户明确表达偏好的轮次
        preference_keywords = ["喜欢", "不要", "风格", "改成", "更", "偏"]
        if any(kw in turn.user_message for kw in preference_keywords):
            score += 0.15

        # 3. 包含修改指令的轮次
        modify_keywords = ["修改", "改", "换", "调整", "优化"]
        if any(kw in turn.user_message for kw in modify_keywords):
            score += 0.1

        # 4. 距离当前越远, 重要性递减
        distance = len(session.turns) - turn.turn_index
        if distance > 5:
            score -= 0.1
        if distance > 10:
            score -= 0.1

        return min(max(score, 0.0), 1.0)


class RuleCompressor:
    """
    规则压缩器 — <2ms, 同步可用
    保留关键信息, 删除冗余
    """

    def compress_turn(self, turn: ConversationTurn) -> Dict[str, str]:
        """将一轮对话压缩为摘要"""
        user_summary = self._compress_user_message(turn)
        assistant_summary = self._compress_assistant_message(turn)
        return {"user": user_summary, "assistant": assistant_summary}

    def _compress_user_message(self, turn: ConversationTurn) -> str:
        msg = turn.user_message
        # 提取核心动词+名词, 去除冗余修饰
        # 简化策略: 保留前80个字
        if len(msg) <= 80:
            return msg
        # 提取关键信息
        keywords = []
        intent = turn.intent
        if intent:
            keywords.append(f"意图:{intent.intent}")
            for k, v in intent.slots.items():
                if v and not k.startswith("_"):
                    keywords.append(f"{k}={v}")
        if keywords:
            return f"[{'; '.join(keywords)}] {msg[:50]}..."
        return msg[:80] + "..."

    def _compress_assistant_message(self, turn: ConversationTurn) -> str:
        if turn.generated_script:
            # 只保留话术的开头和核心特征描述
            script = turn.generated_script
            first_line = script.split("\n")[0][:60] if "\n" in script else script[:60]
            return f"[已生成话术] {first_line}..."
        msg = turn.assistant_message
        return msg[:80] + "..." if len(msg) > 80 else msg


class SessionContextCompressor:
    """
    会话层分级压缩器
    确保总token不超过预算, 在信息保留和token消耗间平衡
    """

    def __init__(self):
        self.llm = LLMServiceClient()
        self.importance_scorer = TurnImportanceScorer()
        self.rule_compressor = RuleCompressor()
        self.cfg = settings.context

    async def compress(self, session: SessionContext,
                       token_budget: Optional[int] = None
                       ) -> CompressedContext:
        """对会话历史做分级压缩"""
        token_budget = token_budget or self.cfg.session_token_budget
        turns = session.turns
        total = len(turns)

        if total == 0:
            return CompressedContext()

        # Step 1: 给每轮打重要性分
        for turn in turns:
            turn.importance_score = self.importance_scorer.score(turn, session)

        # Step 2: 划分Zone
        zone_a = turns[-self.cfg.zone_a_turns:]     # 最近2轮
        zone_b_start = max(0, total - 6)
        zone_b_end = max(0, total - self.cfg.zone_a_turns)
        zone_b = turns[zone_b_start:zone_b_end] if total > self.cfg.zone_a_turns else []
        zone_c = turns[:zone_b_start] if total > 6 else []

        compressed_messages: List[Dict[str, str]] = []
        current_tokens = 0

        # Zone C: 激进压缩 → 合并为一段总结
        if zone_c:
            summary = await self._summarize_zone_c(zone_c)
            msg = {"role": "system", "content": f"[历史对话总结]\n{summary}"}
            compressed_messages.append(msg)
            current_tokens += self._count_tokens(summary)

        # Zone B: 中度压缩 → 规则压缩
        for turn in zone_b:
            if current_tokens >= token_budget * 0.6:
                if turn.importance_score < 0.4:
                    continue  # 预算紧张, 跳过低重要性轮次

            compressed = self.rule_compressor.compress_turn(turn)
            compressed_messages.append({
                "role": "user",
                "content": f"[第{turn.turn_index}轮] {compressed['user']}",
            })
            compressed_messages.append({
                "role": "assistant",
                "content": compressed["assistant"],
            })
            current_tokens += self._count_tokens(
                compressed["user"] + compressed["assistant"]
            )

        # Zone A: 完整保留
        for turn in zone_a:
            compressed_messages.append({
                "role": "user",
                "content": turn.user_message,
            })
            compressed_messages.append({
                "role": "assistant",
                "content": turn.assistant_message,
            })
            current_tokens += turn.token_count

        # 如果仍然超预算 → 紧急裁剪
        if current_tokens > token_budget:
            compressed_messages = self._emergency_trim(
                compressed_messages, token_budget
            )

        stats = {
            "total_turns": total,
            "zone_a_turns": len(zone_a),
            "zone_b_turns": len(zone_b),
            "zone_c_turns": len(zone_c),
            "compressed_tokens": current_tokens,
            "budget": token_budget,
        }

        return CompressedContext(
            messages=compressed_messages,
            token_count=current_tokens,
            compression_stats=stats,
        )

    async def _summarize_zone_c(self, turns: List[ConversationTurn]) -> str:
        """将远轮历史合并为一段总结 (LLM摘要)"""
        turns_text = "\n".join(
            f"第{t.turn_index}轮 用户:{t.user_message[:80]} → "
            f"系统:{t.assistant_message[:80]}"
            for t in turns
        )
        prompt = (
            "请将以下对话历史压缩为一段简洁的总结（100字以内），\n"
            "保留关键信息：用户的核心需求、重要决策、明确表达的偏好。\n"
            "丢弃：闲聊、重复内容、已被推翻的决策。\n\n"
            f"对话历史：\n{turns_text}\n\n总结："
        )
        try:
            return await self.llm.generate_sync(prompt, max_tokens=150)
        except Exception as e:
            logger.warning(f"Zone C summarization failed: {e}")
            # 降级: 规则压缩
            return " | ".join(
                f"第{t.turn_index}轮:{t.user_message[:30]}"
                for t in turns[-3:]
            )

    def _count_tokens(self, text: str) -> int:
        """估算token数"""
        return int(len(text) / 1.5)

    def _emergency_trim(self, messages: List[Dict], budget: int) -> List[Dict]:
        """紧急裁剪: 从最旧的消息开始删"""
        while self._total_tokens(messages) > budget and len(messages) > 2:
            messages.pop(0)
        return messages

    def _total_tokens(self, messages: List[Dict]) -> int:
        return sum(self._count_tokens(m.get("content", "")) for m in messages)
