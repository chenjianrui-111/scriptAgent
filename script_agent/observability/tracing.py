"""
分布式链路追踪 (OpenTelemetry)

为每个请求创建 span, 串联 intent → profile → generate → quality 全链路。
如 opentelemetry 未安装, 使用 NoOp 占位。
"""

import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    _tracer = trace.get_tracer("script_agent")
    _TRACING_AVAILABLE = True
    logger.info("OpenTelemetry tracing initialized")

except ImportError:
    _TRACING_AVAILABLE = False
    logger.info("opentelemetry not installed, tracing disabled")


@contextmanager
def start_span(name: str, trace_id: Optional[str] = None, **attributes):
    """
    创建追踪 span

    Usage:
        with start_span("intent_recognition", trace_id="xxx", query=query):
            result = await agent(msg)
    """
    if not _TRACING_AVAILABLE:
        yield None
        return

    with _tracer.start_as_current_span(name) as span:
        if trace_id:
            span.set_attribute("trace_id", trace_id)
        for k, v in attributes.items():
            if v is not None:
                span.set_attribute(k, str(v) if not isinstance(v, (int, float, bool)) else v)
        try:
            yield span
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
