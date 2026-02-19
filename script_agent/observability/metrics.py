"""
Prometheus 指标采集

核心指标:
  - 请求计数 (按 intent/status)
  - 生成延迟 (histogram)
  - 质量校验结果分布
  - 活跃会话数
  - Skill 路由命中率

如 prometheus_client 未安装, 使用 NoOp 占位, 不影响业务逻辑。
"""

import time
import logging
import threading
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, Info

    APP_INFO = Info(
        "script_agent_app",
        "ScriptAgent app metadata",
    )

    # API层指标
    HTTP_REQUESTS = Counter(
        "script_agent_http_requests_total",
        "HTTP requests count",
        ["method", "path", "status"],
    )
    HTTP_REQUEST_LATENCY = Histogram(
        "script_agent_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "path"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    HTTP_INFLIGHT = Gauge(
        "script_agent_http_inflight_requests",
        "In-flight HTTP requests",
    )

    # 请求指标
    REQUEST_COUNT = Counter(
        "script_agent_requests_total",
        "Total requests",
        ["intent", "status", "skill"],
    )
    REQUEST_LATENCY = Histogram(
        "script_agent_request_duration_seconds",
        "Request duration",
        ["stage"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    # 生成指标
    GENERATION_COUNT = Counter(
        "script_agent_generations_total",
        "Script generations",
        ["category", "scenario"],
    )
    QUALITY_SCORE = Histogram(
        "script_agent_quality_score",
        "Quality check scores",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    QUALITY_RESULT = Counter(
        "script_agent_quality_results_total",
        "Quality check results",
        ["result"],   # passed / failed / retry
    )

    # 系统指标
    ACTIVE_SESSIONS = Gauge(
        "script_agent_active_sessions",
        "Tracked active sessions in current process",
    )
    SESSION_EVENTS = Counter(
        "script_agent_session_events_total",
        "Session lifecycle events",
        ["event"],  # created / deleted
    )
    INTENT_CONFIDENCE = Histogram(
        "script_agent_intent_confidence",
        "Intent classification confidence",
        ["intent"],
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    # Skill 指标
    SKILL_HITS = Counter(
        "script_agent_skill_hits_total",
        "Skill routing hits",
        ["skill_name"],
    )

    LOCK_TIMEOUTS = Counter(
        "script_agent_lock_timeouts_total",
        "Session lock timeout count",
        ["endpoint"],
    )
    WORKFLOW_CACHE_HITS = Counter(
        "script_agent_workflow_cache_hits_total",
        "Workflow dedup cache hit count",
    )
    CHECKPOINT_WRITES = Counter(
        "script_agent_checkpoint_writes_total",
        "Workflow checkpoint writes",
        ["status"],
    )

    _METRICS_AVAILABLE = True
    logger.info("Prometheus metrics initialized")

except ImportError:
    _METRICS_AVAILABLE = False
    logger.info("prometheus_client not installed, metrics disabled")


# ======================================================================
#  便捷 API (安全调用, 不依赖 prometheus_client)
# ======================================================================

_ACTIVE_SESSION_IDS: set[str] = set()
_ACTIVE_SESSIONS_LOCK = threading.Lock()


def set_app_info(name: str, version: str, env: str):
    if _METRICS_AVAILABLE:
        APP_INFO.info({"name": name, "version": version, "env": env})


def observe_http_request(method: str, path: str, status: int, elapsed_seconds: float):
    if _METRICS_AVAILABLE:
        status_str = str(status)
        HTTP_REQUESTS.labels(method=method, path=path, status=status_str).inc()
        HTTP_REQUEST_LATENCY.labels(method=method, path=path).observe(
            max(0.0, elapsed_seconds)
        )


def inc_http_inflight():
    if _METRICS_AVAILABLE:
        HTTP_INFLIGHT.inc()


def dec_http_inflight():
    if _METRICS_AVAILABLE:
        HTTP_INFLIGHT.dec()


def mark_session_created(session_id: str):
    if not _METRICS_AVAILABLE or not session_id:
        return
    SESSION_EVENTS.labels(event="created").inc()
    with _ACTIVE_SESSIONS_LOCK:
        _ACTIVE_SESSION_IDS.add(session_id)
        ACTIVE_SESSIONS.set(len(_ACTIVE_SESSION_IDS))


def mark_session_deleted(session_id: str):
    if not _METRICS_AVAILABLE or not session_id:
        return
    SESSION_EVENTS.labels(event="deleted").inc()
    with _ACTIVE_SESSIONS_LOCK:
        _ACTIVE_SESSION_IDS.discard(session_id)
        ACTIVE_SESSIONS.set(len(_ACTIVE_SESSION_IDS))


def record_request(intent: str, status: str, skill: str = "none"):
    if _METRICS_AVAILABLE:
        REQUEST_COUNT.labels(intent=intent, status=status, skill=skill).inc()


def record_generation(category: str, scenario: str):
    if _METRICS_AVAILABLE:
        GENERATION_COUNT.labels(category=category, scenario=scenario).inc()


def record_quality(score: float, passed: bool):
    if _METRICS_AVAILABLE:
        QUALITY_SCORE.observe(score)
        QUALITY_RESULT.labels(result="passed" if passed else "failed").inc()


def record_intent_confidence(intent: str, confidence: float):
    if _METRICS_AVAILABLE:
        INTENT_CONFIDENCE.labels(intent=intent).observe(confidence)


def record_skill_hit(skill_name: str):
    if _METRICS_AVAILABLE:
        SKILL_HITS.labels(skill_name=skill_name).inc()


def record_lock_timeout(endpoint: str):
    if _METRICS_AVAILABLE:
        LOCK_TIMEOUTS.labels(endpoint=endpoint).inc()


def record_workflow_cache_hit():
    if _METRICS_AVAILABLE:
        WORKFLOW_CACHE_HITS.inc()


def record_checkpoint(status: str):
    if _METRICS_AVAILABLE:
        CHECKPOINT_WRITES.labels(status=status).inc()


def observe_stage_latency_seconds(stage: str, elapsed_seconds: float):
    if _METRICS_AVAILABLE:
        REQUEST_LATENCY.labels(stage=stage).observe(max(0.0, elapsed_seconds))


def observe_stage_latency_ms(stage: str, elapsed_ms: float):
    observe_stage_latency_seconds(stage, elapsed_ms / 1000.0)


@contextmanager
def track_latency(stage: str):
    """上下文管理器 — 自动记录阶段耗时"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if _METRICS_AVAILABLE:
            REQUEST_LATENCY.labels(stage=stage).observe(elapsed)
