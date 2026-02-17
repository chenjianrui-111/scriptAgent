"""
全局配置 - Agent系统运行参数
支持通过环境变量切换 开发/测试/生产 环境
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List


def _load_json_dict(name: str, default: Dict) -> Dict:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def _resolve_qwen_model_by_env(env: str) -> str:
    """开发/测试默认 0.5B，生产默认 7B，可通过环境变量覆盖。"""
    env_key = (env or "").strip().lower()
    if env_key == "production":
        return os.getenv("QWEN_MODEL_PRODUCTION", "qwen2.5:7b")
    return os.getenv("QWEN_MODEL_LOCAL", "qwen2.5:0.5b")


def _default_ollama_model_map(env: str) -> Dict[str, str]:
    model_name = _resolve_qwen_model_by_env(env)
    return {
        "美妆": model_name,
        "食品": model_name,
        "服饰": model_name,
        "通用": model_name,
    }


def _default_primary_backend(env: str) -> str:
    env_key = (env or "").strip().lower()
    if env_key == "production":
        return "vllm"
    return "zhipu"


@dataclass
class LLMConfig:
    """LLM服务配置"""
    # 环境: development / testing / production
    env: str = os.getenv("APP_ENV", "development")

    # 主后端: zhipu | vllm | ollama
    # 默认: 开发/测试 zhipu, 生产 vllm
    primary_backend: str = os.getenv(
        "LLM_PRIMARY_BACKEND",
        _default_primary_backend(os.getenv("APP_ENV", "development")),
    )

    # vLLM (生产环境)
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://vllm-service:8000/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "Qwen/Qwen-7B")

    # Ollama (开发/测试环境)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # LoRA adapter映射 (vLLM: 动态切换 / Ollama: 合并模型名)
    vllm_adapter_map: Dict[str, str] = field(default_factory=lambda: {
        "美妆": "beauty-lora",
        "食品": "food-lora",
        "服饰": "fashion-lora",
        "通用": "Qwen/Qwen-7B",
    })
    ollama_model_map: Dict[str, str] = field(
        default_factory=lambda: _default_ollama_model_map(
            os.getenv("APP_ENV", "development")
        )
    )

    # 生成参数
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    stream: bool = True

    # 失败降级备用模型 (例如千问/Qwen)
    fallback_enabled: bool = (
        os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true"
    )
    fallback_backend: str = os.getenv("LLM_FALLBACK_BACKEND", "ollama")
    fallback_base_url: str = os.getenv(
        "LLM_FALLBACK_BASE_URL",
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    fallback_model: str = os.getenv(
        "LLM_FALLBACK_MODEL",
        _resolve_qwen_model_by_env(os.getenv("APP_ENV", "development")),
    )
    zhipu_base_url: str = os.getenv(
        "ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
    )
    zhipu_api_key: str = os.getenv("ZHIPU_API_KEY", "")
    zhipu_model: str = os.getenv("ZHIPU_MODEL", "glm-4-flash")

    # 话术生成兜底策略
    script_min_chars: int = int(os.getenv("SCRIPT_MIN_CHARS", "40"))
    script_primary_attempts: int = int(os.getenv("SCRIPT_PRIMARY_ATTEMPTS", "2"))

    # 可靠性控制: 重试 / 超时 / 熔断 / 幂等
    retry_max_attempts: int = int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "3"))
    retry_base_delay_seconds: float = float(
        os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "0.35")
    )
    retry_max_delay_seconds: float = float(
        os.getenv("LLM_RETRY_MAX_DELAY_SECONDS", "2.0")
    )
    retry_jitter_seconds: float = float(
        os.getenv("LLM_RETRY_JITTER_SECONDS", "0.2")
    )
    timeout_connect_seconds: float = float(
        os.getenv("LLM_TIMEOUT_CONNECT_SECONDS", "5")
    )
    timeout_read_sync_seconds: float = float(
        os.getenv("LLM_TIMEOUT_READ_SYNC_SECONDS", "25")
    )
    timeout_read_stream_seconds: float = float(
        os.getenv("LLM_TIMEOUT_READ_STREAM_SECONDS", "90")
    )
    timeout_total_sync_seconds: float = float(
        os.getenv("LLM_TIMEOUT_TOTAL_SYNC_SECONDS", "35")
    )
    timeout_total_stream_seconds: float = float(
        os.getenv("LLM_TIMEOUT_TOTAL_STREAM_SECONDS", "120")
    )
    circuit_breaker_enabled: bool = (
        os.getenv("LLM_CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
    )
    circuit_breaker_failure_threshold: int = int(
        os.getenv("LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
    )
    circuit_breaker_recovery_seconds: float = float(
        os.getenv("LLM_CIRCUIT_BREAKER_RECOVERY_SECONDS", "30")
    )
    circuit_breaker_half_open_max_calls: int = int(
        os.getenv("LLM_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS", "1")
    )
    fallback_to_general_enabled: bool = (
        os.getenv("LLM_FALLBACK_TO_GENERAL_ENABLED", "true").lower() == "true"
    )
    fallback_keep_category: bool = (
        os.getenv("LLM_FALLBACK_KEEP_CATEGORY", "false").lower() == "true"
    )
    fallback_timeout_factor: float = float(
        os.getenv("LLM_FALLBACK_TIMEOUT_FACTOR", "0.75")
    )
    idempotency_salt: str = os.getenv("LLM_IDEMPOTENCY_SALT", "script-agent-v1")
    idempotency_inflight_enabled: bool = (
        os.getenv("LLM_IDEMPOTENCY_INFLIGHT_ENABLED", "true").lower() == "true"
    )


@dataclass
class IntentConfig:
    """意图识别配置"""
    enable_ml_classifier: bool = (
        os.getenv("INTENT_ENABLE_ML", "false").lower() == "true"
    )
    # 分级阈值
    high_confidence_threshold: float = 0.85    # Level 1: 快速分类直接采用
    medium_confidence_threshold: float = 0.6   # Level 2: 全量模型验证
    low_confidence_threshold: float = 0.5      # Level 3: LLM兜底
    clarification_threshold: float = 0.5       # 低于此值请求用户澄清

    # 意图类型
    intent_labels: list = field(default_factory=lambda: [
        "script_generation",     # 话术生成
        "script_modification",   # 话术修改
        "script_optimization",   # 话术优化
        "script_translation",    # 话术翻译/风格转换
        "query",                 # 一般查询
        "other",                 # 其他
    ])


@dataclass
class ContextConfig:
    """上下文管理配置"""
    # Token预算分配 (总预算4096)
    total_token_budget: int = 4096
    system_token_budget: int = 500       # 系统层 ~15%
    longterm_token_budget: int = 800     # 长期记忆 ~25%
    session_token_budget: int = 1800     # 会话层 ~60%
    generation_reserve: int = 1000       # 生成预留

    # 压缩Zone划分
    zone_a_turns: int = 2               # 最近N轮完整保留
    zone_b_turns: int = 4               # 中度压缩轮数
    zone_c_min_turns: int = 7           # 超过此轮数进入激进压缩

    # LLMLingua-2 配置
    llmlingua_target_ratio: float = 0.5
    llmlingua_model: str = "xlm-roberta-base"
    # 会话记忆持久化裁剪策略
    max_turns_persisted: int = int(os.getenv("SESSION_MAX_TURNS_PERSISTED", "30"))
    max_scripts_persisted: int = int(
        os.getenv("SESSION_MAX_SCRIPTS_PERSISTED", "80")
    )
    compress_history_on_save: bool = (
        os.getenv("SESSION_COMPRESS_ON_SAVE", "true").lower() == "true"
    )
    compress_message_max_chars: int = int(
        os.getenv("SESSION_COMPRESS_MESSAGE_MAX_CHARS", "120")
    )
    relevance_trim_enabled: bool = (
        os.getenv("SESSION_RELEVANCE_TRIM_ENABLED", "true").lower() == "true"
    )
    relevance_preserve_old_turns: int = int(
        os.getenv("SESSION_RELEVANCE_PRESERVE_OLD_TURNS", "2")
    )
    relevance_query_weight: float = float(
        os.getenv("SESSION_RELEVANCE_QUERY_WEIGHT", "0.45")
    )
    relevance_product_weight: float = float(
        os.getenv("SESSION_RELEVANCE_PRODUCT_WEIGHT", "0.2")
    )
    relevance_intent_weight: float = float(
        os.getenv("SESSION_RELEVANCE_INTENT_WEIGHT", "0.15")
    )
    relevance_recency_weight: float = float(
        os.getenv("SESSION_RELEVANCE_RECENCY_WEIGHT", "0.2")
    )


@dataclass
class QualityConfig:
    """质量校验配置"""
    max_retries: int = 3
    style_consistency_threshold: float = 0.6
    enable_llm_evaluation: bool = False    # LLM综合评估 (高质量场景)


@dataclass
class CacheConfig:
    """缓存配置"""
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # 多级缓存TTL
    local_cache_ttl: int = 300           # L1 本地内存 5min
    redis_cache_ttl: int = 3600          # L2 Redis 1h
    profile_cache_ttl: int = 7200        # 画像缓存 2h


@dataclass
class DomainDataConfig:
    """达人/商品基础数据源配置（SQLite）。"""

    enabled: bool = os.getenv("DOMAIN_DATA_ENABLED", "true").lower() == "true"
    sqlite_path: str = os.getenv("DOMAIN_DATA_DB_PATH", "domain_data.db")
    auto_init_schema: bool = (
        os.getenv("DOMAIN_DATA_AUTO_INIT_SCHEMA", "true").lower() == "true"
    )


@dataclass
class OrchestrationConfig:
    """编排执行配置"""
    session_lock_timeout_seconds: float = float(
        os.getenv("SESSION_LOCK_TIMEOUT_SECONDS", "8")
    )
    checkpoint_auto_save: bool = (
        os.getenv("WORKFLOW_CHECKPOINT_AUTO_SAVE", "true").lower() == "true"
    )
    langgraph_required: bool = (
        os.getenv("LANGGRAPH_REQUIRED", "false").lower() == "true"
    )
    request_dedup_enabled: bool = (
        os.getenv("REQUEST_DEDUP_ENABLED", "true").lower() == "true"
    )
    request_dedup_ttl_seconds: int = int(
        os.getenv("REQUEST_DEDUP_TTL_SECONDS", "120")
    )
    checkpoint_script_max_chars: int = int(
        os.getenv("CHECKPOINT_SCRIPT_MAX_CHARS", "4000")
    )
    distributed_lock_enabled: bool = (
        os.getenv("DISTRIBUTED_LOCK_ENABLED", "true").lower() == "true"
    )
    distributed_lock_prefix: str = os.getenv(
        "DISTRIBUTED_LOCK_PREFIX", "script_agent:lock:session:"
    )
    distributed_lock_lease_seconds: int = int(
        os.getenv("DISTRIBUTED_LOCK_LEASE_SECONDS", "30")
    )
    distributed_lock_retry_interval_ms: int = int(
        os.getenv("DISTRIBUTED_LOCK_RETRY_INTERVAL_MS", "120")
    )


@dataclass
class CheckpointConfig:
    """工作流checkpoint存储配置"""
    store: str = os.getenv("CHECKPOINT_STORE", "memory").lower()
    redis_prefix: str = os.getenv("CHECKPOINT_REDIS_PREFIX", "workflow:checkpoint:")
    sqlite_path: str = os.getenv("CHECKPOINT_DB_PATH", "workflow_checkpoints.db")
    history_limit: int = int(os.getenv("CHECKPOINT_HISTORY_LIMIT", "50"))


@dataclass
class CoreRateLimitConfig:
    """核心接口限流 (QPS + Token)"""
    enabled: bool = os.getenv("CORE_RATE_LIMIT_ENABLED", "true").lower() == "true"
    backend: str = os.getenv("CORE_RATE_LIMIT_BACKEND", "local").lower()
    qps_per_tenant: int = int(os.getenv("CORE_RATE_QPS_PER_TENANT", "8"))
    tokens_per_minute: int = int(os.getenv("CORE_RATE_TOKENS_PER_MIN", "20000"))
    redis_prefix: str = os.getenv("CORE_RATE_REDIS_PREFIX", "script_agent:rate:")


@dataclass
class LongTermMemoryConfig:
    """长期记忆检索配置（Embedding + Vector Search）"""

    enabled: bool = os.getenv("LONGTERM_MEMORY_ENABLED", "true").lower() == "true"
    backend: str = os.getenv("LONGTERM_MEMORY_BACKEND", "memory").lower()
    top_k: int = int(os.getenv("LONGTERM_MEMORY_TOP_K", "3"))
    top_k_min: int = int(os.getenv("LONGTERM_MEMORY_TOP_K_MIN", "2"))
    top_k_max: int = int(os.getenv("LONGTERM_MEMORY_TOP_K_MAX", "8"))
    adaptive_top_k_enabled: bool = (
        os.getenv("LONGTERM_MEMORY_ADAPTIVE_TOPK_ENABLED", "true").lower() == "true"
    )
    min_similarity: float = float(os.getenv("LONGTERM_MEMORY_MIN_SIMILARITY", "0.2"))
    write_back_enabled: bool = (
        os.getenv("LONGTERM_MEMORY_WRITE_BACK_ENABLED", "true").lower() == "true"
    )
    hybrid_enabled: bool = (
        os.getenv("LONGTERM_MEMORY_HYBRID_ENABLED", "true").lower() == "true"
    )
    hybrid_candidate_multiplier: int = int(
        os.getenv("LONGTERM_MEMORY_HYBRID_CANDIDATE_MULTIPLIER", "3")
    )
    hybrid_dense_weight: float = float(
        os.getenv("LONGTERM_MEMORY_HYBRID_DENSE_WEIGHT", "0.65")
    )
    hybrid_sparse_weight: float = float(
        os.getenv("LONGTERM_MEMORY_HYBRID_SPARSE_WEIGHT", "0.35")
    )
    rerank_enabled: bool = (
        os.getenv("LONGTERM_MEMORY_RERANK_ENABLED", "true").lower() == "true"
    )
    rerank_window: int = int(os.getenv("LONGTERM_MEMORY_RERANK_WINDOW", "24"))

    embedding_backend: str = os.getenv(
        "LONGTERM_MEMORY_EMBEDDING_BACKEND", "hash"
    ).lower()
    embedding_dim: int = int(os.getenv("LONGTERM_MEMORY_EMBEDDING_DIM", "256"))
    embedding_model: str = os.getenv(
        "LONGTERM_MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    es_url: str = os.getenv("LONGTERM_MEMORY_ES_URL", "http://localhost:9200")
    es_index: str = os.getenv("LONGTERM_MEMORY_ES_INDEX", "script_agent_memories")
    es_username: str = os.getenv("LONGTERM_MEMORY_ES_USERNAME", "")
    es_password: str = os.getenv("LONGTERM_MEMORY_ES_PASSWORD", "")
    es_timeout_seconds: int = int(os.getenv("LONGTERM_MEMORY_ES_TIMEOUT_SECONDS", "8"))


@dataclass
class ToolSecurityConfig:
    """工具调用安全配置（schema + allowlist + 注入防护）"""

    schema_strict_enabled: bool = (
        os.getenv("TOOL_SCHEMA_STRICT_ENABLED", "true").lower() == "true"
    )
    allowlist_enabled: bool = (
        os.getenv("TOOL_ALLOWLIST_ENABLED", "true").lower() == "true"
    )
    default_role: str = os.getenv("TOOL_DEFAULT_ROLE", "user")
    # 格式: {"user":["script_generation"],"admin":["*"]}
    role_allowlist: Dict[str, List[str]] = field(
        default_factory=lambda: _load_json_dict(
            "TOOL_ROLE_ALLOWLIST_JSON",
            {
                "user": ["script_generation", "script_modification", "batch_generate"],
                "service": ["script_generation", "script_modification", "batch_generate"],
                "admin": ["*"],
            },
        )
    )
    # 格式: {"tenant_demo":["script_generation","script_modification"]}
    tenant_allowlist: Dict[str, List[str]] = field(
        default_factory=lambda: _load_json_dict("TOOL_TENANT_ALLOWLIST_JSON", {})
    )

    prompt_injection_tripwire_enabled: bool = (
        os.getenv("TOOL_PROMPT_TRIPWIRE_ENABLED", "true").lower() == "true"
    )
    prompt_injection_threshold: int = int(
        os.getenv("TOOL_PROMPT_TRIPWIRE_THRESHOLD", "1")
    )
    slot_text_max_chars: int = int(
        os.getenv("TOOL_SLOT_TEXT_MAX_CHARS", "5000")
    )


@dataclass
class AppConfig:
    """应用总配置"""
    app_name: str = "script_agent"
    env: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8080"))

    llm: LLMConfig = field(default_factory=LLMConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    domain_data: DomainDataConfig = field(default_factory=DomainDataConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    core_rate_limit: CoreRateLimitConfig = field(default_factory=CoreRateLimitConfig)
    longterm_memory: LongTermMemoryConfig = field(
        default_factory=LongTermMemoryConfig
    )
    tool_security: ToolSecurityConfig = field(default_factory=ToolSecurityConfig)


# 全局配置单例
settings = AppConfig()
