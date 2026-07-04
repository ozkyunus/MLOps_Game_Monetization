"""
/copilot router — Monetization Copilot (Service 6).

POST /copilot/chat
    Body: { "question": "...", "conversation_id": "..."? }
    Returns: grounded answer + source citations + tool trace + usage.

Agentic RAG over the knowledge base (Qdrant) + platform data tools.
Every turn is persisted to copilot_log (LLMOps telemetry).
"""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException
from sqlmodel import Session

from src.database import engine as db_engine
from src.models import CopilotLog, CopilotRequest

router = APIRouter(prefix="/copilot", tags=["copilot"])


@router.post("/chat")
def copilot_chat(payload: CopilotRequest):
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="Copilot requires GOOGLE_API_KEY (Gemini) — set it in .env",
        )

    from src.copilot.agent import ask  # lazy: keeps module import light
    try:
        result = ask(payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Copilot backend error: {type(exc).__name__}: {exc}",
        ) from exc

    with Session(db_engine) as session:
        session.add(CopilotLog(
            conversation_id=payload.conversation_id,
            question=payload.question,
            answer=result["answer"],
            refused=result["refused"],
            sources=json.dumps(result["sources"], ensure_ascii=False),
            tool_calls=json.dumps(result["tool_trace"], ensure_ascii=False),
            prompt_version=result["prompt_version"],
            model_name=result["model_name"],
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
            latency_ms=result["latency_ms"],
            retrieval_ms=result["retrieval_ms"],
        ))
        session.commit()

    return {
        "question": payload.question,
        "conversation_id": payload.conversation_id,
        **result,
    }
