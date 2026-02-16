"""
风格提取服务 - 从用户采纳的话术中提取风格特征

三管齐下:
  1. 从采纳话术统计分析 (规则+NLP)
  2. 从用户修改对比推断偏好 (diff分析)
  3. 从用户明确表达理解意图 (LLM提取)

更新策略: 加权融合 (新30% + 旧70%, 带时间衰减)
"""

import re
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from script_agent.models.context import (
    SessionContext, StyleProfile, ConversationTurn,
)

logger = logging.getLogger(__name__)


@dataclass
class CatchphraseResult:
    phrase: str
    category: str
    frequency: int
    confidence: float


@dataclass
class ExtractedStyle:
    """提取出的风格特征"""
    tone: str = ""
    formality_level: float = 0.5
    catchphrases: List[str] = field(default_factory=list)
    avg_sentence_length: float = 20.0
    punctuation_style: str = "normal"
    selling_point_style: str = "direct"
    interaction_frequency: float = 0.5
    humor_level: float = 0.3
    sample_count: int = 0
    confidence: float = 0.5


class CatchphraseExtractor:
    """口头禅提取器"""

    KNOWN_PATTERNS = {
        "addressing": [
            r"姐妹们?", r"宝子们?", r"家人们?", r"宝宝们?",
            r"小仙女们?", r"集美们?", r"兄弟们?", r"朋友们?",
        ],
        "emphasis": [
            r"真的", r"绝了", r"爱了", r"yyds", r"绝绝子",
        ],
        "recommendation": [
            r"必入", r"必买", r"闭眼入", r"强推", r"安利",
        ],
        "modal": [
            r"啊+", r"呀+", r"哦+", r"嘿+", r"哈+",
        ],
    }

    def __init__(self):
        self.compiled = {
            cat: [re.compile(p) for p in patterns]
            for cat, patterns in self.KNOWN_PATTERNS.items()
        }

    def extract(self, texts: List[str]) -> List[CatchphraseResult]:
        all_matches: List[Tuple[str, str]] = []
        for text in texts:
            for cat, patterns in self.compiled.items():
                for pat in patterns:
                    for m in pat.finditer(text):
                        all_matches.append((m.group(), cat))

        freq = Counter(phrase for phrase, _ in all_matches)
        results = []
        for phrase, count in freq.most_common(20):
            if count >= 2 or (texts and count / len(texts) >= 0.3):
                cat = next(
                    (c for p, c in all_matches if p == phrase), "other"
                )
                results.append(CatchphraseResult(
                    phrase=phrase, category=cat,
                    frequency=count,
                    confidence=min(count / max(len(texts), 1), 1.0),
                ))
        return results


class FormalityAnalyzer:
    """正式度分析"""

    CASUAL_MARKERS = ["啊", "呀", "哦", "嘿", "哈", "！", "~", "哇", "嘻"]
    FORMAL_MARKERS = ["尊敬的", "您好", "特此", "建议您", "值得关注"]

    def analyze(self, text: str) -> float:
        """返回正式度 0(口语) - 1(正式)"""
        casual = sum(text.count(m) for m in self.CASUAL_MARKERS)
        formal = sum(text.count(m) for m in self.FORMAL_MARKERS)
        total = max(casual + formal, 1)
        return formal / total


class StyleExtractor:
    """风格特征提取 - 综合规则+NLP"""

    def __init__(self):
        self.catchphrase_extractor = CatchphraseExtractor()
        self.formality_analyzer = FormalityAnalyzer()

    def extract_from_texts(self, texts: List[str]) -> ExtractedStyle:
        """从多个采纳话术中提取风格"""
        if not texts:
            return ExtractedStyle()

        # 1. 口头禅
        catchphrases = self.catchphrase_extractor.extract(texts)
        top_phrases = [cp.phrase for cp in catchphrases[:5]]

        # 2. 正式度
        formality_scores = [self.formality_analyzer.analyze(t) for t in texts]
        avg_formality = sum(formality_scores) / len(formality_scores)

        # 3. 句子长度
        all_sentences = []
        for text in texts:
            sents = re.split(r'[。！？~\n]', text)
            all_sentences.extend(s for s in sents if len(s) > 2)
        avg_sent_len = (
            sum(len(s) for s in all_sentences) / len(all_sentences)
            if all_sentences else 20.0
        )

        # 4. 标点风格
        total_chars = sum(len(t) for t in texts)
        exclaim_count = sum(t.count("！") + t.count("!") for t in texts)
        wave_count = sum(t.count("~") for t in texts)
        if total_chars > 0:
            exclaim_ratio = exclaim_count / total_chars
            punct_style = (
                "enthusiastic" if exclaim_ratio > 0.03
                else "calm" if exclaim_ratio < 0.005
                else "normal"
            )
        else:
            punct_style = "normal"

        # 5. 互动频率
        interaction_markers = ["?", "？", "吗", "呢", "点赞", "关注"]
        interaction_count = sum(
            sum(t.count(m) for m in interaction_markers) for t in texts
        )
        interaction_freq = min(interaction_count / max(len(texts), 1) / 5, 1.0)

        # 6. 幽默度 (简化: 看emoji和搞笑词汇频率)
        humor_markers = ["哈哈", "笑", "😂", "🤣", "搞笑", "段子"]
        humor_count = sum(
            sum(t.count(m) for m in humor_markers) for t in texts
        )
        humor_level = min(humor_count / max(len(texts), 1) / 3, 1.0)

        # 7. 语气推断
        if avg_formality < 0.3:
            tone = "活泼口语"
        elif avg_formality > 0.7:
            tone = "正式专业"
        else:
            tone = "自然亲和"

        return ExtractedStyle(
            tone=tone,
            formality_level=avg_formality,
            catchphrases=top_phrases,
            avg_sentence_length=avg_sent_len,
            punctuation_style=punct_style,
            interaction_frequency=interaction_freq,
            humor_level=humor_level,
            sample_count=len(texts),
            confidence=min(0.5 + len(texts) * 0.05, 0.95),
        )


class ProfileUpdater:
    """
    画像更新器 - 加权融合策略
    新提取30% + 现有画像70% (带时间衰减)
    """

    NEW_WEIGHT = 0.3
    EXISTING_WEIGHT = 0.7

    def merge(self, existing: StyleProfile,
              extracted: ExtractedStyle) -> StyleProfile:
        """融合新提取的风格到现有画像"""
        w_new = self.NEW_WEIGHT
        w_old = self.EXISTING_WEIGHT

        merged = StyleProfile(
            influencer_id=existing.influencer_id,
            tone=extracted.tone if extracted.confidence > existing.confidence else existing.tone,
            formality_level=round(
                existing.formality_level * w_old + extracted.formality_level * w_new, 3
            ),
            catchphrases=self._merge_catchphrases(
                existing.catchphrases, extracted.catchphrases
            ),
            avg_sentence_length=round(
                existing.avg_sentence_length * w_old + extracted.avg_sentence_length * w_new, 1
            ),
            punctuation_style=extracted.punctuation_style,
            interaction_frequency=round(
                existing.interaction_frequency * w_old + extracted.interaction_frequency * w_new, 3
            ),
            humor_level=round(
                existing.humor_level * w_old + extracted.humor_level * w_new, 3
            ),
            sample_count=existing.sample_count + extracted.sample_count,
            confidence=min(existing.confidence + 0.05, 0.95),
            last_updated=datetime.now(),
            version=existing.version + 1,
        )
        return merged

    def _merge_catchphrases(self, old: List[str],
                             new: List[str]) -> List[str]:
        """合并口头禅, 保留频率最高的"""
        seen = set()
        result = []
        for phrase in new + old:
            if phrase not in seen:
                seen.add(phrase)
                result.append(phrase)
        return result[:10]
