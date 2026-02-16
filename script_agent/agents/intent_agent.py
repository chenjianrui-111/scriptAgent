"""
意图识别Agent - 分级策略: 高置信快返回, 低置信LLM兜底

三级分级:
  Level 1: 小模型快速分类 (DistilBERT) ~10ms, confidence >= 0.85 → 直接使用
  Level 2: 全量模型验证 (BERT) ~80ms, 0.6 <= confidence < 0.85
  Level 3: LLM兜底 ~500ms, confidence < 0.6; 若 < 0.5 → 请求用户澄清
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional

from script_agent.agents.base import BaseAgent
from script_agent.models.message import AgentMessage, IntentResult
from script_agent.models.context import SessionContext
from script_agent.services.llm_client import LLMServiceClient
from script_agent.config.settings import settings

logger = logging.getLogger(__name__)


# ========================================================================
#  槽位提取器
# ========================================================================

class SlotExtractor:
    """
    槽位提取: 规则优先, NER补充
    槽位类型固定, 模式可枚举, 规则覆盖率高且可解释可调试
    """

    # 品类关键词 + 同义词扩展
    CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "美妆": ["美妆", "化妆品", "护肤", "彩妆", "口红", "面膜", "精华"],
        "食品": ["食品", "零食", "美食", "吃的", "食材", "饮品"],
        "服饰": ["服饰", "穿搭", "衣服", "女装", "男装", "鞋包"],
        "数码": ["数码", "电子", "手机", "电脑", "耳机", "数码产品"],
    }

    # 达人名提取正则
    NAME_PATTERN = re.compile(
        r"(?:给|帮|为|替)(.{2,8}?)(?:达人|主播|博主|老师|生成|写|创作)"
    )

    EVENT_KEYWORDS: Dict[str, str] = {
        "618": "618大促", "双11": "双十一", "双十一": "双十一",
        "年货节": "年货节", "38节": "38女王节", "双12": "双十二",
    }
    PRODUCT_NAME_PATTERNS = [
        re.compile(r"(?:商品|产品|单品|款式)[:：\s]*([\u4e00-\u9fffA-Za-z0-9\-]{2,24})(?:的|，|,|\s|$)"),
        re.compile(r"(?:这款|这个|该)([\u4e00-\u9fffA-Za-z0-9\-]{2,24}?)(?:的|，|,|\s|$)"),
    ]

    def extract(self, query: str, intent: str = "") -> Dict[str, Any]:
        slots: Dict[str, Any] = {}

        # 1. 品类识别
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                slots["category"] = cat
                break

        # 2. 达人名提取
        match = self.NAME_PATTERN.search(query)
        if match:
            name = match.group(1).strip()
            slots["target_name"] = name
            # 实际项目中: slots["target_id"] = self._lookup_id(name)

        # 3. 场景识别
        if "直播" in query:
            slots["scenario"] = "直播带货"
            if "开场" in query:
                slots["sub_scenario"] = "开场话术"
            elif "产品" in query or "介绍" in query:
                slots["sub_scenario"] = "产品介绍"
            elif "促销" in query or "秒杀" in query:
                slots["sub_scenario"] = "促销话术"
        elif "短视频" in query or "视频" in query:
            slots["scenario"] = "短视频"
        elif "种草" in query:
            slots["scenario"] = "种草文案"

        # 4. 活动识别
        for kw, event in self.EVENT_KEYWORDS.items():
            if kw in query:
                slots["event"] = event
                break

        # 5. 风格提示
        style_hints = []
        style_map = {
            "活泼": "lively", "专业": "professional", "幽默": "humorous",
            "亲和": "friendly", "高端": "premium", "口语": "casual",
        }
        for zh, en in style_map.items():
            if zh in query:
                style_hints.append(en)
        if style_hints:
            slots["style_hint"] = ",".join(style_hints)

        # 6. 商品槽位（商品名/卖点）
        for pattern in self.PRODUCT_NAME_PATTERNS:
            match = pattern.search(query)
            if match:
                slots["product_name"] = match.group(1).strip()
                break
        if "卖点" in query:
            # e.g. 卖点：成分安全、持妆久
            for sep in ("卖点:", "卖点："):
                if sep in query:
                    raw = query.split(sep, 1)[1]
                    points = [
                        p.strip()
                        for p in re.split(r"[，,、;；。\n]", raw)
                        if p.strip()
                    ]
                    if points:
                        slots["selling_points"] = points[:8]
                    break

        return slots


# ========================================================================
#  指代消解器
# ========================================================================

class ReferenceResolver:
    """
    三类指代的统一解析:
    1. 实体指代: "她"/"这个达人" → EntityCache
    2. 内容指代: "那段话"/"第一版" → 生成历史
    3. 隐式继承: "再来一段"/"继续" → 继承上轮槽位
    """

    ENTITY_PATTERNS = [
        (re.compile(r"(她|他|这个达人|那个主播|这位)"), "influencer", "target_name"),
        (re.compile(r"(这个商品|那个产品|这款)"), "product", "product_name"),
    ]
    CONTENT_PATTERNS = [
        re.compile(r"(那段|那个|上次那个|第[一二三四五]版|第[一二三四五]次)"),
    ]
    CONTINUATION_PATTERNS = re.compile(
        r"(再来一段|继续|再写一个|换一个|再生成)"
    )

    def resolve(self, query: str,
                session: SessionContext) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}

        # 1. 实体指代
        for pattern, entity_type, slot_name in self.ENTITY_PATTERNS:
            if pattern.search(query):
                cached = session.entity_cache.get_latest(entity_type)
                if cached:
                    resolved[slot_name] = cached.get("name", "")
                    resolved[f"{slot_name}_id"] = cached.get("id", "")

        # 2. 内容指代
        for pat in self.CONTENT_PATTERNS:
            m = pat.search(query)
            if m and session.generated_scripts:
                # 简单策略: 取最近一条生成内容
                resolved["target_content_id"] = session.generated_scripts[-1].script_id

        # 3. 隐式继承
        if self.CONTINUATION_PATTERNS.search(query):
            resolved.update(session.slot_context.slots)
            resolved["_continuation"] = True

        return resolved


# ========================================================================
#  意图澄清器
# ========================================================================

class IntentClarifier:
    """槽位缺失时的主动澄清"""

    REQUIRED_SLOTS: Dict[str, List[str]] = {
        "script_generation": ["category", "scenario"],
        "script_modification": ["target_content_id"],
    }

    CLARIFY_TEMPLATES: Dict[str, str] = {
        "category": "请问您想生成哪个品类的话术？（美妆/食品/服饰/数码/其他）",
        "scenario": "请问是用于直播带货还是短视频？",
        "target_content_id": "请问您想修改哪一段内容？",
    }

    def check(self, intent: str, slots: Dict) -> Optional[str]:
        required = self.REQUIRED_SLOTS.get(intent, [])
        missing = [s for s in required if s not in slots or slots[s] is None]
        if not missing:
            return None
        return self.CLARIFY_TEMPLATES.get(missing[0], f"请提供更多信息: {missing[0]}")


# ========================================================================
#  意图识别Agent主类
# ========================================================================

# ========================================================================
#  意图分类器 (TF-IDF + jieba + LogisticRegression)
# ========================================================================

class IntentClassifier:
    """
    意图分类器 — TF-IDF + LogisticRegression

    相比关键词匹配:
      - 支持同义词泛化 (jieba 分词后向量化)
      - 输出校准的概率 (而非命中数 * 0.3)
      - 可通过新增训练数据持续改进

    训练数据内置, 也支持从文件加载扩展。
    如有 scikit-learn 则使用 ML 分类, 否则降级为加权关键词匹配。
    """

    # 内置训练数据 — 格式: (文本, 意图标签)
    TRAINING_DATA: List[tuple] = [
        # script_generation
        ("帮我写一段美妆直播开场话术", "script_generation"),
        ("生成一段食品种草文案", "script_generation"),
        ("帮我创作一段直播话术", "script_generation"),
        ("写一段促销话术", "script_generation"),
        ("来一段服饰介绍话术", "script_generation"),
        ("帮我做一段直播带货脚本", "script_generation"),
        ("给我生成一段短视频文案", "script_generation"),
        ("写一段产品介绍话术", "script_generation"),
        ("帮我创建一段618促销话术", "script_generation"),
        ("写个直播开场白", "script_generation"),
        ("生成美妆产品种草内容", "script_generation"),
        ("帮这个达人写一段话术", "script_generation"),
        ("再来一段活泼风格的", "script_generation"),
        ("写一段幽默的直播话术", "script_generation"),
        ("我需要一段带货话术", "script_generation"),

        # script_modification
        ("把这段话术改一下", "script_modification"),
        ("修改上面那段话术", "script_modification"),
        ("帮我改改第一版", "script_modification"),
        ("把开头换成问候语", "script_modification"),
        ("不要这个风格换一个", "script_modification"),
        ("替换掉里面的口头禅", "script_modification"),
        ("把语气改成专业一点", "script_modification"),
        ("这段太长了帮我缩短", "script_modification"),
        ("加一些互动引导", "script_modification"),
        ("把那个产品名换成面膜", "script_modification"),

        # script_optimization
        ("帮我润色一下这段话术", "script_optimization"),
        ("优化一下语言表达", "script_optimization"),
        ("让这段话术更有感染力", "script_optimization"),
        ("提升一下专业度", "script_optimization"),
        ("加强互动性", "script_optimization"),
        ("调整一下节奏感", "script_optimization"),
        ("让它更口语化一些", "script_optimization"),
        ("增强说服力", "script_optimization"),

        # query
        ("什么是直播话术", "query"),
        ("怎么写好种草文案", "query"),
        ("美妆直播有什么技巧", "query"),
        ("618活动什么时候开始", "query"),
        ("目前支持哪些品类", "query"),
        ("你能做什么", "query"),
    ]

    def __init__(self):
        self._ml_available = False
        self._vectorizer = None
        self._classifier = None
        self._keyword_fallback = self._build_keyword_weights()
        if settings.intent.enable_ml_classifier:
            self._try_init_ml()
        else:
            logger.info(
                "IntentClassifier: ML mode disabled by INTENT_ENABLE_ML, "
                "using weighted-keyword fallback."
            )

    def _try_init_ml(self):
        """尝试加载 scikit-learn, 失败则降级为关键词匹配"""
        try:
            import jieba
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            texts = [t[0] for t in self.TRAINING_DATA]
            labels = [t[1] for t in self.TRAINING_DATA]

            # jieba 分词
            tokenized = [" ".join(jieba.cut(t)) for t in texts]

            self._vectorizer = TfidfVectorizer(max_features=500)
            X = self._vectorizer.fit_transform(tokenized)

            self._classifier = LogisticRegression(
                max_iter=200, multi_class="multinomial",
            )
            self._classifier.fit(X, labels)
            self._ml_available = True
            logger.info("IntentClassifier: ML mode (TF-IDF + LogisticRegression)")
        except ImportError:
            logger.info("IntentClassifier: Fallback mode (weighted keywords). "
                        "Install scikit-learn + jieba for ML classification.")
        except Exception as e:  # pragma: no cover - 运行环境依赖异常降级
            logger.warning(
                "IntentClassifier: ML init failed (%s), fallback to keywords.",
                e,
            )

    def _build_keyword_weights(self) -> Dict[str, Dict[str, float]]:
        """加权关键词表 (降级方案, 比原始版本更精确)"""
        return {
            "script_generation": {
                "帮我写": 0.4, "写一段": 0.4, "生成": 0.35, "创作": 0.35,
                "来一段": 0.35, "做一段": 0.3, "创建": 0.3, "写": 0.2,
                "给我": 0.15, "一段": 0.1, "话术": 0.15, "文案": 0.15,
            },
            "script_modification": {
                "修改": 0.4, "改一下": 0.4, "替换": 0.35, "换成": 0.35,
                "不要这个": 0.3, "改": 0.25, "换": 0.2, "缩短": 0.25,
                "加一些": 0.2,
            },
            "script_optimization": {
                "润色": 0.45, "优化": 0.4, "提升": 0.35, "更好": 0.3,
                "加强": 0.3, "调整": 0.25, "增强": 0.3, "更有": 0.2,
                "感染力": 0.2,
            },
            "query": {
                "什么是": 0.4, "怎么": 0.35, "有什么": 0.3, "技巧": 0.25,
                "你能": 0.3, "支持": 0.2,
            },
        }

    def classify(self, query: str) -> tuple:
        """
        分类主入口

        Returns:
            (intent, confidence)
        """
        if self._ml_available:
            return self._ml_classify(query)
        return self._keyword_classify(query)

    def _ml_classify(self, query: str) -> tuple:
        """ML 分类 (TF-IDF + LR)"""
        import jieba
        tokenized = " ".join(jieba.cut(query))
        X = self._vectorizer.transform([tokenized])
        proba = self._classifier.predict_proba(X)[0]
        classes = self._classifier.classes_
        best_idx = proba.argmax()
        return classes[best_idx], float(proba[best_idx])

    def _keyword_classify(self, query: str) -> tuple:
        """加权关键词分类 (降级方案)"""
        scores: Dict[str, float] = {}
        for intent, weights in self._keyword_fallback.items():
            score = sum(w for kw, w in weights.items() if kw in query)
            scores[intent] = min(score, 0.95)

        if not scores or max(scores.values()) < 0.1:
            return "script_generation", 0.4

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best, scores[best]


class IntentRecognitionAgent(BaseAgent):
    """
    意图识别Agent
    分级策略: 小模型(TF-IDF+LR) → LLM兜底
    """

    def __init__(self):
        super().__init__(name="intent_recognition")
        self.slot_extractor = SlotExtractor()
        self.reference_resolver = ReferenceResolver()
        self.clarifier = IntentClarifier()
        self.llm = LLMServiceClient()
        self.cfg = settings.intent
        self.classifier = IntentClassifier()

    async def process(self, message: AgentMessage) -> AgentMessage:
        query = message.payload.get("query", "")
        session: SessionContext = message.payload.get("session", SessionContext())

        # Step 1: 快速分类 (模拟小模型)
        intent, confidence = self._fast_classify(query)

        # Step 2: 分级判断
        if confidence < self.cfg.medium_confidence_threshold:
            # 低置信 → LLM深度理解
            llm_result = await self._llm_intent_recognition(query, session)
            intent = llm_result.get("intent", intent)
            confidence = llm_result.get("confidence", confidence)

        # Step 3: 槽位提取
        slots = self.slot_extractor.extract(query, intent)

        # Step 4: 指代消解 (多轮对话关键)
        resolved = self.reference_resolver.resolve(query, session)
        slots.update(resolved)

        # Step 5: 上下文补全
        slots = self._fill_from_context(slots, session)

        # Step 6: 检查是否需要澄清
        clarification = None
        if confidence >= self.cfg.clarification_threshold:
            clarification = self.clarifier.check(intent, slots)

        intent_result = IntentResult(
            intent=intent,
            confidence=confidence,
            slots=slots,
            needs_clarification=clarification is not None,
            clarification_question=clarification or "",
        )

        return message.create_response(
            payload={"intent_result": intent_result},
            source=self.name,
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _fast_classify(self, query: str) -> tuple:
        """快速分类 (TF-IDF + LogisticRegression / 加权关键词降级)"""
        return self.classifier.classify(query)

    def _fill_from_context(self, slots: Dict, session: SessionContext) -> Dict:
        """从历史对话补全缺失槽位"""
        # 从上轮继承 category
        if "category" not in slots and session.slot_context.slots.get("category"):
            slots["category"] = session.slot_context.slots["category"]
            slots["_category_source"] = "context_fill"

        # 从上轮继承 scenario
        if "scenario" not in slots and session.slot_context.slots.get("scenario"):
            slots["scenario"] = session.slot_context.slots["scenario"]

        # 从上轮继承 target_name
        if "target_name" not in slots:
            cached = session.entity_cache.get_latest("influencer")
            if cached:
                slots["target_name"] = cached.get("name", "")

        # 从历史继承商品信息
        if "product_name" not in slots and session.slot_context.slots.get("product_name"):
            slots["product_name"] = session.slot_context.slots["product_name"]
        if "product_name" not in slots:
            cached_product = session.entity_cache.get_latest("product")
            if cached_product:
                slots["product_name"] = cached_product.get("name", "")

        return slots

    async def _llm_intent_recognition(self, query: str,
                                       session: SessionContext) -> Dict:
        """LLM兜底意图识别"""
        history_str = "\n".join(
            f"用户: {t.user_message}" for t in session.turns[-3:]
        ) if session.turns else "无"

        prompt = f"""你是一个意图识别专家。请分析用户输入，提取意图和槽位。

## 用户输入
{query}

## 历史对话
{history_str}

## 输出格式 (JSON)
{{
    "intent": "script_generation|script_modification|script_optimization|other",
    "confidence": 0.0-1.0,
    "slots": {{
        "category": "美妆|食品|服饰|数码|null",
        "target_name": "达人名字或null",
        "scenario": "直播带货|短视频|种草文案|null",
        "requirements": "其他具体要求"
    }},
    "reasoning": "简要推理过程"
}}

请直接输出JSON，不要包含其他内容。
"""
        try:
            raw = await self.llm.generate_sync(prompt, max_tokens=300)
            # 尝试解析JSON
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].strip("json\n ")
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"LLM intent recognition failed: {e}")
            return {"intent": "script_generation", "confidence": 0.5, "slots": {}}
