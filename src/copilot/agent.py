"""Monetization Copilot — agentic RAG core.

Design (see docs/specs/2026-07-03-copilot-rag-llmops-design.md):
  - Gemini with bound tools; an EXPLICIT tool loop (max MAX_TOOL_ROUNDS)
    instead of an opaque agent framework — full control over telemetry,
    tracing and guardrails, and nothing to fight when versions move.
  - `search_knowledge_base` is a tool like any other: the model decides
    per-question whether it needs documents, platform data, or both.
  - Every run returns a structured result: answer + citations + tool trace
    + token usage + prompt version. The router persists it to copilot_log.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src.copilot import vectorstore

load_dotenv()

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "copilot_v2.md"
PROMPT_VERSION = PROMPT_PATH.stem.split("_")[-1]          # "v1"
MODEL_NAME = "gemini-2.5-flash-lite"
MAX_TOOL_ROUNDS = 4
REFUSAL_MARKER = "Bu, platformun kapsamı dışında"


# ── Tools ────────────────────────────────────────────────────────────────────
# Each tool returns a JSON string (the model reads it as text) and appends a
# record to the per-request trace via the closure set up in `ask()`.

@tool
def search_knowledge_base(query: str) -> str:
    """Search industry benchmark / policy / platform documentation.
    Use for questions about industry numbers, best practices, store
    policies, or how this platform and its models work."""
    chunks = vectorstore.search(query)
    return json.dumps(
        [{k: c[k] for k in ("doc_id", "title", "section", "text")} for c in chunks],
        ensure_ascii=False,
    )


@tool
def get_channel_roi() -> str:
    """Get the user's OWN acquisition data: per-channel CPI, observed ROAS,
    payback estimate and blended totals, from the platform database."""
    from src.routers.channel import channel_roi
    return json.dumps(channel_roi(), ensure_ascii=False, default=str)


@tool
def get_retention(dim: str = "channel") -> str:
    """Get the user's OWN cohort retention proxies (D1/D7/D30) grouped by
    `dim` = 'channel', 'country' or 'platform'."""
    from src.routers.cohort import cohort_retention
    return json.dumps(cohort_retention(dim=dim), ensure_ascii=False, default=str)


@tool
def predict_pltv(user_id: str) -> str:
    """Predict payer probability and pLTV for a single user_id from the
    platform's two-tower model."""
    from src.ml.inference import predict_pltv as _predict
    result = _predict(user_id)
    result.pop("raw_features", None)          # DataFrame — not serializable
    return json.dumps(result, ensure_ascii=False, default=str)


@tool
def draft_offer_copy(user_id: str, context: str = "app_open") -> str:
    """Generate IAP offer copy (title/body/CTA) for a user_id in a given
    context: 'level_complete', 'after_loss' or 'app_open'."""
    from src.models import DecisionRequest
    from src.routers.personalized import personalized_offer
    payload = DecisionRequest(user_id=user_id, context=context)
    return json.dumps(personalized_offer(payload), ensure_ascii=False, default=str)


TOOLS = [search_knowledge_base, get_channel_roi, get_retention,
         predict_pltv, draft_offer_copy]
_TOOLS_BY_NAME = {t.name: t for t in TOOLS}


@lru_cache(maxsize=1)
def _llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME, temperature=0.2, timeout=45,
    ).bind_tools(TOOLS)


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    return PROMPT_PATH.read_text()


def ask(question: str) -> dict:
    """Run one copilot turn. Returns a structured result dict."""
    t0 = time.perf_counter()
    messages = [SystemMessage(content=_system_prompt()),
                HumanMessage(content=question)]

    tool_trace: list[dict] = []
    sources: dict[str, dict] = {}             # doc_id → citation (deduped)
    usage = {"input_tokens": 0, "output_tokens": 0}
    retrieval_ms = 0.0

    response = None
    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = _llm().invoke(messages)
        meta = getattr(response, "usage_metadata", None) or {}
        usage["input_tokens"] += meta.get("input_tokens", 0)
        usage["output_tokens"] += meta.get("output_tokens", 0)

        if not getattr(response, "tool_calls", None):
            break

        messages.append(response)
        for call in response.tool_calls:
            name, args = call["name"], call.get("args", {})
            t_tool = time.perf_counter()
            try:
                result = _TOOLS_BY_NAME[name].invoke(args)
            except Exception as exc:
                result = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            elapsed = (time.perf_counter() - t_tool) * 1000

            if name == "search_knowledge_base":
                retrieval_ms += elapsed
                for chunk in json.loads(result):
                    sources[chunk["doc_id"]] = {
                        "doc_id": chunk["doc_id"],
                        "title": chunk["title"],
                        "section": chunk["section"],
                    }
            tool_trace.append({"tool": name, "args": args,
                               "ms": round(elapsed, 1)})
            messages.append(ToolMessage(content=result,
                                        tool_call_id=call["id"]))

    # Gemini may return content as a list of blocks — flatten to plain text.
    content = response.content
    if isinstance(content, list):
        answer = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    else:
        answer = str(content)
    return {
        "answer": answer,
        "refused": answer.strip().startswith(REFUSAL_MARKER),
        "sources": list(sources.values()),
        "tool_trace": tool_trace,
        "usage": usage,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "retrieval_ms": round(retrieval_ms, 1),
        "prompt_version": PROMPT_VERSION,
        "model_name": MODEL_NAME,
    }
