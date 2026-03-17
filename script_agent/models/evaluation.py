"""
评估体系数据模型

参考 Anthropic "Demystifying Evals for AI Agents" 方法论:
  - GoldenExample: 黄金数据集单条样本
  - JudgeVerdict: LLM-as-Judge 五维度结构化评分
  - EvalResult: 单条评估结果
  - EvalReport: 评估报告汇总
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GoldenExample:
    """黄金数据集单条样本"""
    example_id: str = ""
    # 输入
    input_query: str = ""
    category: str = ""                          # 美妆 / 食品 / 服饰 / 通用
    scenario: str = ""                          # 直播 / 短视频
    tags: List[str] = field(default_factory=list)  # normal / product_switch / style / modify / adversarial / edge

    # 上下文
    influencer_name: str = ""
    influencer_style: str = ""                  # 活泼 / 专业 / 温柔
    product_name: str = ""
    product_info: str = ""
    previous_product_name: str = ""             # 商品切换场景

    # 期望
    expected_keywords: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    expected_min_score: float = 0.7
    dimension_thresholds: Dict[str, float] = field(default_factory=dict)

    # 元数据
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeVerdict:
    """LLM-as-Judge 五维度结构化评分"""
    content_accuracy: float = 0.0       # 内容准确性 (卖点覆盖、信息正确)
    language_quality: float = 0.0       # 语言质量 (流畅度、口语自然度)
    style_match: float = 0.0            # 风格匹配 (与达人画像一致性)
    structure_completeness: float = 0.0  # 结构完整性 (开头→卖点→互动→收尾)
    compliance_safety: float = 0.0      # 合规安全 (无敏感词、无违规承诺)

    overall_score: float = 0.0
    reasoning: str = ""
    raw_response: str = ""

    def compute_weighted_score(self) -> float:
        """加权综合评分"""
        score = (
            0.25 * self.content_accuracy
            + 0.20 * self.language_quality
            + 0.20 * self.style_match
            + 0.20 * self.structure_completeness
            + 0.15 * self.compliance_safety
        )
        self.overall_score = round(score, 4)
        return self.overall_score


@dataclass
class EvalResult:
    """单条评估结果"""
    example_id: str = ""
    run_index: int = 0                          # pass@k 中的第几次运行
    generated_content: str = ""
    quality_score: float = 0.0                  # QualityAgent 规则检测分
    judge_verdict: Optional[JudgeVerdict] = None
    passed: bool = False
    error: str = ""
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CategoryBreakdown:
    """按品类/场景/标签的细分统计"""
    name: str = ""
    total: int = 0
    passed: int = 0
    avg_score: float = 0.0
    avg_dimensions: Dict[str, float] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


@dataclass
class EvalReport:
    """评估报告汇总"""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    # 总体指标
    total_examples: int = 0
    total_runs: int = 0
    pass_count: int = 0
    pass_rate: float = 0.0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0

    # pass@k 指标
    k: int = 1
    pass_at_k: float = 0.0                     # 至少一次通过概率

    # 维度均分
    avg_dimensions: Dict[str, float] = field(default_factory=dict)

    # 细分
    by_category: List[CategoryBreakdown] = field(default_factory=list)
    by_tag: List[CategoryBreakdown] = field(default_factory=list)

    # 详细结果
    results: List[EvalResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
