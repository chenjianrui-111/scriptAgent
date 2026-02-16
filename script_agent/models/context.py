"""
会话与上下文数据模型
三层上下文: 系统层(System) + 会话层(Session) + 长期记忆层(LongTerm)
"""

import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from script_agent.models.message import IntentResult, GeneratedScript


# ===================================================================
#  系统层上下文 (System Context)
#  生命周期: 应用级，常驻不变
#  Token占比: ~15% (~500 tokens)
# ===================================================================

@dataclass
class SystemContext:
    """系统层上下文 - 应用级常驻配置"""
    base_role: str = ""
    category_knowledge: str = ""
    compliance_rules: str = ""
    output_format: str = ""
    tenant_config: str = ""
    prompt: str = ""                   # 拼装后的完整System Prompt

    @property
    def token_count(self) -> int:
        """估算token数 (中文约1.5字/token)"""
        return int(len(self.prompt) / 1.5)

    @property
    def prefix_hash(self) -> str:
        """Prefix Caching key - 同品类+同场景+同租户 共享前缀"""
        return hashlib.md5(self.prompt.encode()).hexdigest()


# ===================================================================
#  会话层上下文 (Session Context)
#  生命周期: 单次会话
#  Token占比: ~60% (需要分级压缩)
# ===================================================================

@dataclass
class Token:
    """分词结果"""
    word: str
    pos: str  # 词性


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    turn_index: int = 0
    # 用户侧
    user_message: str = ""               # 用户原始输入
    resolved_message: str = ""           # 指代消解后
    intent: Optional[IntentResult] = None
    # 系统侧
    assistant_message: str = ""
    generated_script: Optional[str] = None
    # 元信息
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    # 压缩相关
    summary: Optional[str] = None
    is_compressed: bool = False
    importance_score: float = 0.5


@dataclass
class EntityCache:
    """
    实体缓存 - 用于多轮对话指代消解
    记录当前操作实体，按类型分slot存储
    """
    entities: Dict[str, Any] = field(default_factory=dict)

    def update(self, entity_type: str, entity_id: str,
               entity_name: str = ""):
        self.entities[entity_type] = {
            "id": entity_id,
            "name": entity_name,
            "updated_at": datetime.now().isoformat(),
        }

    def get_latest(self, entity_type: str) -> Optional[Dict]:
        return self.entities.get(entity_type)


@dataclass
class SlotContext:
    """槽位上下文 - 跨轮保留槽位信息"""
    slots: Dict[str, Any] = field(default_factory=dict)
    last_intent: str = ""

    def update(self, intent: str, slots: Dict[str, Any]):
        self.last_intent = intent
        self.slots.update({k: v for k, v in slots.items() if v is not None})


@dataclass
class SessionContext:
    """会话层上下文 - 单次会话的完整状态"""
    # 基础信息
    session_id: str = ""
    tenant_id: str = ""
    influencer_id: str = ""
    influencer_name: str = ""
    category: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # 对话历史 (核心, 需要压缩管理)
    turns: List[ConversationTurn] = field(default_factory=list)

    # 实体状态 (指代消解依赖)
    entity_cache: EntityCache = field(default_factory=EntityCache)
    slot_context: SlotContext = field(default_factory=SlotContext)

    # 生成内容记录
    generated_scripts: List[GeneratedScript] = field(default_factory=list)

    # 状态机
    current_state: str = "INIT"
    state_history: List[str] = field(default_factory=list)

    def add_turn(self, user_message: str, assistant_message: str,
                 intent: Optional[IntentResult] = None,
                 generated_script: Optional[str] = None) -> ConversationTurn:
        """添加一轮对话"""
        turn = ConversationTurn(
            turn_index=len(self.turns),
            user_message=user_message,
            assistant_message=assistant_message,
            intent=intent,
            generated_script=generated_script,
            token_count=int((len(user_message) + len(assistant_message)) / 1.5),
        )
        self.turns.append(turn)
        return turn


@dataclass
class CompressedContext:
    """压缩后的上下文"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    token_count: int = 0
    compression_stats: Dict[str, Any] = field(default_factory=dict)


# ===================================================================
#  长期记忆层 (Long-term Memory)
#  生命周期: 跨会话持久化, 与达人绑定
#  Token占比: ~25% (检索后注入)
# ===================================================================

@dataclass
class StyleProfile:
    """达人风格画像"""
    influencer_id: str = ""
    # 基础风格
    tone: str = ""                           # 语气: 活泼/专业/亲和
    formality_level: float = 0.5             # 正式度 0-1
    catchphrases: List[str] = field(default_factory=list)  # 口头禅
    # 定量特征
    avg_sentence_length: float = 20.0
    punctuation_style: str = "normal"
    selling_point_style: str = "direct"
    interaction_frequency: float = 0.5
    humor_level: float = 0.3
    # 元信息
    sample_count: int = 0
    confidence: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)
    version: int = 1


@dataclass
class InfluencerProfile:
    """达人完整画像"""
    influencer_id: str = ""
    name: str = ""
    # 基本信息
    category: str = ""
    platform: str = ""
    follower_count: int = 0
    # 风格画像
    style: StyleProfile = field(default_factory=StyleProfile)
    # 受众画像
    audience_age_range: str = ""
    audience_gender_ratio: str = ""
    # 历史优质内容
    top_content_keywords: List[str] = field(default_factory=list)
    # embedding (向量化, 用于相似话术检索)
    embedding: Optional[List[float]] = None


@dataclass
class SessionSummary:
    """会话总结 (持久化到长期记忆)"""
    session_id: str = ""
    influencer_id: str = ""
    turn_count: int = 0
    scripts_generated: int = 0
    scripts_adopted: int = 0
    summary: str = ""
    topics_covered: List[str] = field(default_factory=list)
    scenarios_used: List[str] = field(default_factory=list)
    style_insights: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
