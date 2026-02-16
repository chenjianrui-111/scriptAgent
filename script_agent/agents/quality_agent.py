"""
质量校验Agent - 后置校验, 多维度并行检测

检测维度:
  1. 敏感词检测 (AC自动机/DFA)
  2. 合规检测 (广告法极限词、虚假宣传)
  3. 风格一致性 (与画像style匹配度)
  4. LLM综合评估 (可选, 高质量场景)

校验结果: passed=true → COMPLETED
          passed=false + retry<3 → REGENERATING (带反馈重生成)
"""

import asyncio
import logging
import re
from typing import Dict, List, Tuple

from script_agent.agents.base import BaseAgent
from script_agent.models.message import AgentMessage, GeneratedScript, QualityResult
from script_agent.models.context import InfluencerProfile, StyleProfile
from script_agent.services.llm_client import LLMServiceClient
from script_agent.config.settings import settings

logger = logging.getLogger(__name__)


class AhoCorasickAutomaton:
    """
    Aho-Corasick 自动机 — 多模式匹配, O(n + m + z) 复杂度

    n = 文本长度, m = 模式总长度, z = 匹配数
    相比逐词 `in` 检测, 当词库 > 50 条时性能优势明显。
    """

    def __init__(self):
        self._goto: List[Dict[str, int]] = [{}]   # goto 函数
        self._fail: List[int] = [0]                # fail 指针
        self._output: List[List[tuple]] = [[]]     # 输出 (word, category)
        self._built = False

    def add_word(self, word: str, category: str):
        """添加模式串"""
        state = 0
        for ch in word:
            if ch not in self._goto[state]:
                new_state = len(self._goto)
                self._goto.append({})
                self._fail.append(0)
                self._output.append([])
                self._goto[state][ch] = new_state
            state = self._goto[state][ch]
        self._output[state].append((word, category))
        self._built = False

    def build(self):
        """构建 fail 指针 (BFS)"""
        from collections import deque
        queue = deque()
        # 初始化: 深度1节点的 fail 指向根
        for ch, s in self._goto[0].items():
            self._fail[s] = 0
            queue.append(s)
        # BFS 构建 fail
        while queue:
            r = queue.popleft()
            for ch, s in self._goto[r].items():
                queue.append(s)
                state = self._fail[r]
                while state != 0 and ch not in self._goto[state]:
                    state = self._fail[state]
                self._fail[s] = self._goto[state].get(ch, 0)
                if self._fail[s] == s:
                    self._fail[s] = 0
                self._output[s] = self._output[s] + self._output[self._fail[s]]
        self._built = True

    def search(self, text: str) -> List[tuple]:
        """
        搜索文本中的所有匹配

        Returns:
            [(position, word, category), ...]
        """
        if not self._built:
            self.build()
        results = []
        state = 0
        for i, ch in enumerate(text):
            while state != 0 and ch not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(ch, 0)
            for word, category in self._output[state]:
                results.append((i - len(word) + 1, word, category))
        return results


class SensitiveWordChecker:
    """
    敏感词检测 — Aho-Corasick 自动机 + 上下文白名单

    白名单机制: 某些包含敏感词的短语在特定上下文中是合规的,
    例如 "第一次使用" 中的 "第一" 不应触发。
    """

    SENSITIVE_WORDS = {
        "极限词": [
            "最好", "最佳", "第一", "唯一", "首个", "全网最低",
            "绝对", "100%", "永久", "万能",
        ],
        "虚假宣传": [
            "国家级", "驰名商标", "免检", "包治百病",
            "根治", "特效", "秘方",
        ],
        "违禁词": [
            "假货", "山寨", "三无产品",
        ],
    }

    # 上下文白名单: 包含这些模式的匹配不视为违规
    WHITELIST_PATTERNS = [
        re.compile(r"第一次"),           # "第一次使用" 中的"第一"
        re.compile(r"第一[步个件]"),      # "第一步" "第一个"
        re.compile(r"最好的选择之一"),    # "之一" 修饰后不绝对
        re.compile(r"最好[看吃用]的之一"),
        re.compile(r"做到最好"),          # 表达愿望
        re.compile(r"追求最[好佳]"),
    ]

    def __init__(self):
        self._automaton = AhoCorasickAutomaton()
        for category, words in self.SENSITIVE_WORDS.items():
            for word in words:
                self._automaton.add_word(word, category)
        self._automaton.build()

    def check(self, text: str) -> Tuple[bool, List[Dict]]:
        """
        Returns:
            (passed, issues)
        """
        matches = self._automaton.search(text)
        issues = []
        for pos, word, category in matches:
            if self._is_whitelisted(text, pos, word):
                continue
            issues.append({
                "type": "sensitive_word",
                "category": category,
                "word": word,
                "position": pos,
                "suggestion": f"请替换'{word}'为合规表达",
            })
        return len(issues) == 0, issues

    def _is_whitelisted(self, text: str, pos: int, word: str) -> bool:
        """检查匹配是否在白名单上下文中"""
        # 取匹配位置前后的上下文窗口
        start = max(0, pos - 3)
        end = min(len(text), pos + len(word) + 5)
        context = text[start:end]
        for pattern in self.WHITELIST_PATTERNS:
            if pattern.search(context):
                return True
        return False


class ComplianceChecker:
    """合规检测 - 广告法/平台规范"""

    PATTERNS = [
        # 绝对化用语
        (re.compile(r"(最|第一|唯一|首个|全球领先)"), "绝对化用语"),
        # 虚假承诺
        (re.compile(r"(保证.*效果|承诺.*天见效|无效退款)"), "虚假承诺"),
        # 价格欺诈
        (re.compile(r"(原价\d+.*现价|骨折价|跳楼价)"), "价格欺诈嫌疑"),
        # 诱导行为
        (re.compile(r"(不买.*后悔|错过.*没有|最后\d+件)"), "过度诱导"),
    ]

    def check(self, text: str) -> Tuple[bool, List[Dict]]:
        issues = []
        for pattern, issue_type in self.PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                issues.append({
                    "type": "compliance",
                    "category": issue_type,
                    "matched": match if isinstance(match, str) else match[0] if match else "",
                    "suggestion": f"请检查并修改{issue_type}相关表达",
                })
        return len(issues) == 0, issues


class StyleConsistencyChecker:
    """风格一致性检测 - 与达人画像对比"""

    def check(self, text: str, profile: InfluencerProfile) -> Tuple[float, List[str]]:
        """
        Returns:
            (consistency_score, suggestions)
        """
        style = profile.style
        score = 0.7   # 基准分
        suggestions = []

        # 1. 口头禅检查
        if style.catchphrases:
            has_catchphrase = any(cp in text for cp in style.catchphrases)
            if has_catchphrase:
                score += 0.1
            else:
                suggestions.append(
                    f"建议加入达人口头禅: {', '.join(style.catchphrases[:3])}"
                )

        # 2. 正式度检查
        casual_markers = ["啊", "呀", "哦", "嘿", "哈", "！", "~"]
        casual_count = sum(text.count(m) for m in casual_markers)
        text_len = max(len(text), 1)
        actual_casual = casual_count / text_len

        if style.formality_level < 0.4 and actual_casual < 0.02:
            suggestions.append("话术偏正式，建议增加口语化表达")
            score -= 0.05
        elif style.formality_level > 0.7 and actual_casual > 0.05:
            suggestions.append("话术偏口语，建议适当提升专业度")
            score -= 0.05

        # 3. 句子长度检查
        sentences = re.split(r'[。！？~\n]', text)
        sentences = [s for s in sentences if len(s) > 2]
        if sentences:
            avg_len = sum(len(s) for s in sentences) / len(sentences)
            if abs(avg_len - style.avg_sentence_length) > 10:
                suggestions.append(
                    f"句子平均长度({avg_len:.0f}字)与达人习惯"
                    f"({style.avg_sentence_length:.0f}字)差异较大"
                )
                score -= 0.05

        # 4. 互动性检查
        interaction_markers = ["?", "？", "吗", "呢", "点赞", "关注", "评论"]
        has_interaction = any(m in text for m in interaction_markers)
        if style.interaction_frequency > 0.5 and not has_interaction:
            suggestions.append("建议增加互动引导（提问/点赞引导）")
            score -= 0.05

        return min(max(score, 0.0), 1.0), suggestions


class QualityCheckAgent(BaseAgent):
    """
    质量校验Agent
    多维度并行检测, 综合评分决定是否通过
    """

    def __init__(self):
        super().__init__(name="quality_check")
        self.sensitive_checker = SensitiveWordChecker()
        self.compliance_checker = ComplianceChecker()
        self.style_checker = StyleConsistencyChecker()
        self.llm = LLMServiceClient()
        self.cfg = settings.quality

    async def process(self, message: AgentMessage) -> AgentMessage:
        script: GeneratedScript = message.payload.get("script", GeneratedScript())
        profile: InfluencerProfile = message.payload.get(
            "profile", InfluencerProfile()
        )
        text = script.content

        # 并行执行多维度检测
        sensitive_task = asyncio.to_thread(self.sensitive_checker.check, text)
        compliance_task = asyncio.to_thread(self.compliance_checker.check, text)
        style_task = asyncio.to_thread(self.style_checker.check, text, profile)

        (sensitive_passed, sensitive_issues), \
        (compliance_passed, compliance_issues), \
        (style_score, style_suggestions) = await asyncio.gather(
            sensitive_task, compliance_task, style_task
        )

        # 可选: LLM综合评估 (高质量场景)
        llm_score = 0.8
        if self.cfg.enable_llm_evaluation:
            llm_score = await self._llm_evaluate(text, profile)

        # 综合评分
        all_issues = sensitive_issues + compliance_issues
        overall_score = self._calculate_overall_score(
            sensitive_passed, compliance_passed, style_score, llm_score
        )
        passed = (
            sensitive_passed
            and compliance_passed
            and style_score >= self.cfg.style_consistency_threshold
        )

        quality_result = QualityResult(
            passed=passed,
            overall_score=overall_score,
            sensitive_words=[i["word"] for i in sensitive_issues],
            compliance_issues=[i["category"] for i in compliance_issues],
            style_consistency=style_score,
            suggestions=style_suggestions + [i["suggestion"] for i in all_issues],
        )

        # 更新script质量分
        script.quality_score = overall_score

        return message.create_response(
            payload={
                "quality_result": quality_result,
                "script": script,
                "all_issues": all_issues,
            },
            source=self.name,
        )

    def _calculate_overall_score(self, sensitive_ok: bool,
                                  compliance_ok: bool,
                                  style_score: float,
                                  llm_score: float) -> float:
        """综合评分 (加权)"""
        score = 0.0
        score += 0.3 * (1.0 if sensitive_ok else 0.0)     # 敏感词30%
        score += 0.3 * (1.0 if compliance_ok else 0.0)     # 合规30%
        score += 0.2 * style_score                          # 风格20%
        score += 0.2 * llm_score                            # LLM评估20%
        return round(score, 3)

    async def _llm_evaluate(self, text: str,
                             profile: InfluencerProfile) -> float:
        """LLM综合评估"""
        prompt = f"""请对以下话术进行质量评估，打分0-1：

话术内容：
{text[:500]}

评估维度：
1. 内容质量（是否有吸引力、逻辑通顺）
2. 风格匹配（是否符合{profile.style.tone}风格）
3. 专业度（品类知识是否准确）
4. 互动性（是否有互动引导）

请只输出一个0到1之间的分数，不要输出其他内容。
"""
        try:
            result = await self.llm.generate_sync(prompt, max_tokens=10)
            return float(result.strip())
        except (ValueError, Exception):
            return 0.75  # 默认分
