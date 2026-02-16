"""
全局配置 - Agent系统运行参数
支持通过环境变量切换 开发/测试/生产 环境
"""

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class LLMConfig:
    """LLM服务配置"""
    # 环境: development / testing / production
    env: str = os.getenv("APP_ENV", "development")

    # vLLM (生产环境)
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://vllm-service:8000/v1")
    vllm_model: str = "Qwen/Qwen-7B"

    # Ollama (开发/测试环境)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # LoRA adapter映射 (vLLM: 动态切换 / Ollama: 合并模型名)
    vllm_adapter_map: Dict[str, str] = field(default_factory=lambda: {
        "美妆": "beauty-lora",
        "食品": "food-lora",
        "服饰": "fashion-lora",
        "通用": "Qwen/Qwen-7B",
    })
    ollama_model_map: Dict[str, str] = field(default_factory=lambda: {
        "美妆": "qwen-beauty:7b",
        "食品": "qwen-food:7b",
        "服饰": "qwen-fashion:7b",
        "通用": "qwen:7b",
    })

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
    fallback_model: str = os.getenv("LLM_FALLBACK_MODEL", "qwen:7b")


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
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    core_rate_limit: CoreRateLimitConfig = field(default_factory=CoreRateLimitConfig)


# 全局配置单例
settings = AppConfig()
