"""Prometheus metrics — ops metrics via the instrumentator + custom LLM
metrics for the Copilot (the Grafana "LLM dashboard" reads these).

Custom metrics are module-level singletons; the copilot router records into
them per turn. Prometheus scrapes /metrics (ServiceMonitor, 15s).
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

COPILOT_TURNS = Counter(
    "copilot_turns_total",
    "Copilot chat turns served",
    ["refused"],                       # "true" / "false"
)

COPILOT_TOKENS = Counter(
    "copilot_tokens_total",
    "LLM tokens consumed by the copilot (cost driver)",
    ["direction"],                     # "input" / "output"
)

COPILOT_TOOL_CALLS = Counter(
    "copilot_tool_calls_total",
    "Tool invocations by the copilot agent",
    ["tool"],
)

COPILOT_LATENCY = Histogram(
    "copilot_latency_seconds",
    "End-to-end copilot turn latency",
    buckets=(1, 2, 4, 8, 15, 30, 60),
)

COPILOT_RETRIEVAL_LATENCY = Histogram(
    "copilot_retrieval_seconds",
    "Qdrant retrieval time within a copilot turn",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2),
)


def record_copilot_turn(result: dict) -> None:
    """Record one copilot turn's telemetry into Prometheus metrics."""
    COPILOT_TURNS.labels(refused=str(result["refused"]).lower()).inc()
    COPILOT_TOKENS.labels(direction="input").inc(result["usage"]["input_tokens"])
    COPILOT_TOKENS.labels(direction="output").inc(result["usage"]["output_tokens"])
    for call in result["tool_trace"]:
        COPILOT_TOOL_CALLS.labels(tool=call["tool"]).inc()
    COPILOT_LATENCY.observe(result["latency_ms"] / 1000)
    if result["retrieval_ms"]:
        COPILOT_RETRIEVAL_LATENCY.observe(result["retrieval_ms"] / 1000)
