"""
Observability, Structured JSON Logging, Intent vs. Outcome Capture & OpenTelemetry Tracing
(core/observability.py)
Covers Category 4 of the Google ADK Architecture — - Structured JSON Logging
- Intent vs. Outcome Capture
- Distributed Tracing via OpenTelemetry
- Integrated with PIIRedactingLogFilter
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional

from core.pii_redaction import PIIRedactingLogFilter, scrub_pii

# Context variables for distributed trace propagation across async/sync boundaries
_CURRENT_TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_trace_id", default=""
)
_CURRENT_SPAN_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_span_id", default=""
)
_SPAN_STACK: contextvars.ContextVar[List[Dict[str, Any]]] = contextvars.ContextVar(
    "span_stack", default=[]
)

# Attempt optional OpenTelemetry SDK binding; provide full OpenTelemetry-compliant
# span recorder so distributed traces work seamlessly in all environments.
try:
    from opentelemetry import trace as otel_trace  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import (  # type: ignore
        SimpleSpanProcessor,
        InMemorySpanExporter,
    )

    _OTEL_PROVIDER = TracerProvider()
    _OTEL_EXPORTER = InMemorySpanExporter()
    _OTEL_PROVIDER.add_span_processor(SimpleSpanProcessor(_OTEL_EXPORTER))
    otel_trace.set_tracer_provider(_OTEL_PROVIDER)
    _OTEL_TRACER = otel_trace.get_tracer("endurance_copilot.agentops")
    HAS_OTEL = True
except Exception:
    _OTEL_TRACER = None
    _OTEL_EXPORTER = None
    HAS_OTEL = False


# Ring buffer of recent exported spans and intent/outcome audit records for inspection & evaluation
_RECENT_SPANS: List[Dict[str, Any]] = []
_RECENT_INTENT_OUTCOME_PAIRS: List[Dict[str, Any]] = []
_MAX_BUFFER_SIZE = 500


class StructuredJsonFormatter(logging.Formatter):
    """
    Structured JSON Logging
    Formats Python LogRecords as single-line Google Cloud Logging compatible JSON payloads
    with severity, trace_id, span_id, agent metadata, and PII scrubbing.
    """

    def format(self, record: logging.LogRecord) -> str:
        trace_id = getattr(record, "trace_id", None) or _CURRENT_TRACE_ID.get() or "no-trace"
        span_id = getattr(record, "span_id", None) or _CURRENT_SPAN_ID.get() or "no-span"

        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": scrub_pii(record.getMessage()),
            "logging.googleapis.com/trace": trace_id,
            "logging.googleapis.com/spanId": span_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "module": record.module,
            "function": record.funcName,
        }

        for extra_field in (
            "event_type",
            "intent_id",
            "agent_name",
            "tool_name",
            "model_routed",
            "intended_action",
            "intended_arguments",
            "actual_outcome",
            "outcome_status",
            "latency_ms",
            "metadata",
        ):
            if hasattr(record, extra_field):
                payload[extra_field] = scrub_pii(getattr(record, extra_field))

        if record.exc_info:
            payload["exception"] = scrub_pii(self.formatException(record.exc_info))

        return json.dumps(payload, ensure_ascii=False, default=str)


def get_structured_logger(name: str = "EnduranceCopilot.ADK") -> logging.Logger:
    """
    Returns a logger configured with `StructuredJsonFormatter` and `PIIRedactingLogFilter`.
    """
    logger = logging.getLogger(name)
    if not any(isinstance(f, PIIRedactingLogFilter) for f in logger.filters):
        logger.addFilter(PIIRedactingLogFilter())

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        handler.addFilter(PIIRedactingLogFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


obs_logger = get_structured_logger("EnduranceCopilot.Observability")


# -----------------------------------------------------------------------------
# Distributed Tracing (OpenTelemetry Spans & Trace Linking)
# -----------------------------------------------------------------------------

def get_current_trace_id() -> str:
    """Returns active request trace_id or initializes a new W3C trace ID."""
    tid = _CURRENT_TRACE_ID.get()
    if not tid:
        tid = uuid.uuid4().hex
        _CURRENT_TRACE_ID.set(tid)
    return tid


def get_current_span_id() -> str:
    """Returns active span_id."""
    return _CURRENT_SPAN_ID.get() or uuid.uuid4().hex[:16]


@contextmanager
def start_trace_span(
    span_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Creates a linked distributed tracing span (backed by OpenTelemetry when installed
    plus an inspectable span buffer) linking query ingress -> routing -> agent -> tool -> answer.
    """
    parent_span_id = _CURRENT_SPAN_ID.get() or None
    if trace_id:
        active_trace_id = trace_id
    elif parent_span_id is None:
        active_trace_id = uuid.uuid4().hex
    else:
        active_trace_id = get_current_trace_id()
    span_id = uuid.uuid4().hex[:16]

    token_trace = _CURRENT_TRACE_ID.set(active_trace_id)
    token_span = _CURRENT_SPAN_ID.set(span_id)

    clean_attrs = scrub_pii(attributes or {})
    span_record: Dict[str, Any] = {
        "trace_id": active_trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "span_name": span_name,
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "attributes": clean_attrs,
        "status": "OK",
    }
    t0 = time.perf_counter()

    otel_cm = (
        _OTEL_TRACER.start_as_current_span(span_name)
        if HAS_OTEL and _OTEL_TRACER is not None
        else None
    )
    otel_span = otel_cm.__enter__() if otel_cm is not None else None
    if otel_span is not None:
        for k, v in clean_attrs.items():
            try:
                otel_span.set_attribute(str(k), str(v))
            except Exception:
                pass

    try:
        yield span_record
    except Exception as exc:
        span_record["status"] = "ERROR"
        span_record["error"] = scrub_pii(str(exc))
        if otel_span is not None:
            try:
                otel_span.set_attribute("error", True)
                otel_span.set_attribute("error.message", str(exc))
            except Exception:
                pass
        raise
    finally:
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        span_record["duration_ms"] = duration_ms
        span_record["end_time_utc"] = datetime.now(timezone.utc).isoformat()
        _RECENT_SPANS.append(span_record)
        if len(_RECENT_SPANS) > _MAX_BUFFER_SIZE:
            del _RECENT_SPANS[: len(_RECENT_SPANS) - _MAX_BUFFER_SIZE]

        if otel_cm is not None:
            try:
                otel_cm.__exit__(None, None, None)
            except Exception:
                pass

        _CURRENT_SPAN_ID.reset(token_span)
        _CURRENT_TRACE_ID.reset(token_trace)


def get_exported_spans(trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns captured OpenTelemetry spans, optionally filtered by `trace_id`."""
    if trace_id:
        return [s for s in _RECENT_SPANS if s.get("trace_id") == trace_id]
    return list(_RECENT_SPANS)


# -----------------------------------------------------------------------------
# Intent vs. Outcome Capture
# -----------------------------------------------------------------------------

@contextmanager
def capture_intent_and_outcome(
    agent_name: str,
    intended_action: str,
    intended_arguments: Optional[Dict[str, Any]] = None,
    rationale: str = "",
) -> Generator[Dict[str, Any], None, None]:
    """
    Intent vs. Outcome Capture
    Explicitly logs the agent's INTENDED action BEFORE execution (`AGENT_INTENT_PRE_EXECUTION`)
    and the ACTUAL outcome AFTER execution (`AGENT_OUTCOME_POST_EXECUTION`), linking both
    via `intent_id` and `trace_id`.
    """
    intent_id = f"intent_{uuid.uuid4().hex[:12]}"
    trace_id = get_current_trace_id()
    clean_args = scrub_pii(intended_arguments or {})

    # 1. Pre-Execution Intent Log
    intent_record: Dict[str, Any] = {
        "intent_id": intent_id,
        "trace_id": trace_id,
        "agent_name": agent_name,
        "intended_action": intended_action,
        "intended_arguments": clean_args,
        "rationale": scrub_pii(rationale),
        "pre_execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "outcome_recorded": False,
        "actual_outcome": None,
        "outcome_status": "PENDING",
    }

    obs_logger.info(
        f"[INTENT] {agent_name} intends to execute '{intended_action}'",
        extra={
            "event_type": "AGENT_INTENT_PRE_EXECUTION",
            "intent_id": intent_id,
            "agent_name": agent_name,
            "intended_action": intended_action,
            "intended_arguments": clean_args,
            "metadata": {"rationale": rationale},
        },
    )

    t0 = time.perf_counter()
    try:
        with start_trace_span(
            f"{agent_name}.{intended_action}",
            attributes={
                "intent_id": intent_id,
                "agent.name": agent_name,
                "tool.name": intended_action,
            },
        ):
            yield intent_record

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        intent_record["outcome_recorded"] = True
        if intent_record["outcome_status"] == "PENDING":
            intent_record["outcome_status"] = "SUCCESS"
        intent_record["latency_ms"] = latency_ms
        intent_record["post_execution_timestamp"] = datetime.now(timezone.utc).isoformat()
        intent_record["actual_outcome"] = scrub_pii(intent_record.get("actual_outcome"))

        # 2. Post-Execution Outcome Log
        obs_logger.info(
            f"[OUTCOME] {agent_name} completed '{intended_action}' with status={intent_record['outcome_status']}",
            extra={
                "event_type": "AGENT_OUTCOME_POST_EXECUTION",
                "intent_id": intent_id,
                "agent_name": agent_name,
                "intended_action": intended_action,
                "outcome_status": intent_record["outcome_status"],
                "actual_outcome": intent_record["actual_outcome"],
                "latency_ms": latency_ms,
            },
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        intent_record["outcome_recorded"] = True
        intent_record["outcome_status"] = "ERROR"
        intent_record["actual_outcome"] = {"error": scrub_pii(str(exc))}
        intent_record["latency_ms"] = latency_ms
        intent_record["post_execution_timestamp"] = datetime.now(timezone.utc).isoformat()

        obs_logger.error(
            f"[OUTCOME_ERROR] {agent_name} failed '{intended_action}': {exc}",
            extra={
                "event_type": "AGENT_OUTCOME_POST_EXECUTION",
                "intent_id": intent_id,
                "agent_name": agent_name,
                "intended_action": intended_action,
                "outcome_status": "ERROR",
                "actual_outcome": intent_record["actual_outcome"],
                "latency_ms": latency_ms,
            },
        )
        raise
    finally:
        _RECENT_INTENT_OUTCOME_PAIRS.append(intent_record)
        if len(_RECENT_INTENT_OUTCOME_PAIRS) > _MAX_BUFFER_SIZE:
            del _RECENT_INTENT_OUTCOME_PAIRS[: len(_RECENT_INTENT_OUTCOME_PAIRS) - _MAX_BUFFER_SIZE]

        # Also persist to SQLite audit_log when DB is available
        try:
            from core.db import append_audit_event
            append_audit_event("INTENT_OUTCOME_TRACE", intent_record)
        except Exception:
            pass


def trace_agent_action(agent_name: str, action_name: Optional[str] = None) -> Callable:
    """
    Decorator that wraps any agent method or tool in both an OpenTelemetry span
    and an Intent-vs-Outcome audit record.
    """
    def decorator(func: Callable) -> Callable:
        resolved_action = action_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with capture_intent_and_outcome(
                agent_name=agent_name,
                intended_action=resolved_action,
                intended_arguments=kwargs,
                rationale=f"Executing {agent_name}.{resolved_action}",
            ) as rec:
                result = func(*args, **kwargs)
                rec["actual_outcome"] = (
                    result if isinstance(result, (dict, list, str, int, float, bool)) else str(result)
                )
                if isinstance(result, dict) and result.get("status") == "error":
                    rec["outcome_status"] = "GUIDED_RECOVERY_ERROR"
                return result

        return wrapper

    return decorator


def get_intent_outcome_audit_log(limit: int = 50) -> List[Dict[str, Any]]:
    """Returns recent Intent vs. Outcome capture records for verification and observability."""
    return list(reversed(_RECENT_INTENT_OUTCOME_PAIRS[-limit:]))
