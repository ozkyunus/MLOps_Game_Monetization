"""Qdrant access for the Copilot knowledge base.

One collection ("knowledge"), 768-dim Gemini text-embedding-004 vectors.
Everything is lazy so importing this module needs neither Qdrant nor an API
key (CI-safe — same pattern as the DB engines).
"""
from __future__ import annotations

import os
from functools import lru_cache

COLLECTION = "knowledge"
# text-embedding-004 was retired; gemini-embedding-001 is the current stable
# embedder (3072-dim default, normalized).
EMBED_MODEL = "models/gemini-embedding-001"
VECTOR_SIZE = 3072
DEFAULT_TOP_K = 4


@lru_cache(maxsize=1)
def get_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))


@lru_cache(maxsize=1)
def get_embedder():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Embed the query and return the nearest knowledge chunks."""
    vector = get_embedder().embed_query(query)
    hits = get_client().query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        {
            "doc_id":  h.payload.get("doc_id"),
            "title":   h.payload.get("title"),
            "section": h.payload.get("section"),
            "url":     h.payload.get("url"),
            "text":    h.payload.get("text"),
            "score":   round(float(h.score), 4),
        }
        for h in hits
    ]
