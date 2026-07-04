"""Ingest the knowledge/ corpus into Qdrant (Copilot RAG knowledge base).

    uv run python -m scripts.ingest_knowledge

Chunking: one chunk per `## ` section (heading-based — sections are written
to be self-contained, see knowledge/_TEMPLATE.md). Each chunk carries source
metadata for citations. Idempotent: deterministic UUIDv5 point ids + full
collection recreate per run; a corpus manifest (file hashes) is stored so
runs are traceable.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # stable ns


def parse_doc(path: Path) -> dict | None:
    """Parse front-matter + body; returns None for non-corpus files."""
    text = path.read_text()
    match = re.match(r"(?s)^(?:<!--.*?-->\s*)?---\n(.*?)\n---\n(.*)$", text)
    if not match:
        return None
    meta_block, body = match.groups()
    meta = {}
    for line in meta_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return {"meta": meta, "body": body, "doc_id": path.stem}


def chunk_doc(doc: dict) -> list[dict]:
    """Split body on ## headings; every chunk = heading + its text."""
    parts = re.split(r"(?m)^## ", doc["body"])
    chunks = []
    for part in parts[1:]:                     # parts[0] = preamble/title
        section, _, text = part.partition("\n")
        text = text.strip()
        if not text:
            continue
        chunks.append({
            "doc_id": doc["doc_id"],
            "title": doc["meta"].get("title", doc["doc_id"]),
            "section": section.strip(),
            "url": doc["meta"].get("url", ""),
            "source": doc["meta"].get("source", ""),
            "text": f"{section.strip()}\n\n{text}",
        })
    return chunks


def main() -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from src.copilot.vectorstore import (
        COLLECTION,
        VECTOR_SIZE,
        get_client,
        get_embedder,
    )

    docs, corpus_hash = [], hashlib.sha256()
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        if path.name.startswith("_"):
            continue
        doc = parse_doc(path)
        if doc is None:
            print(f"  ⚠ {path.name}: no front-matter — skipped")
            continue
        docs.append(doc)
        corpus_hash.update(path.read_bytes())

    chunks = [c for d in docs for c in chunk_doc(d)]
    print(f"Corpus: {len(docs)} docs → {len(chunks)} chunks "
          f"(hash {corpus_hash.hexdigest()[:12]})")

    print("Embedding (Gemini gemini-embedding-001)...")
    vectors = get_embedder().embed_documents([c["text"] for c in chunks])

    client = get_client()
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid5(NAMESPACE, f"{c['doc_id']}|{c['section']}")),
                vector=v,
                payload=c,
            )
            for c, v in zip(chunks, vectors, strict=True)
        ],
    )
    info = client.get_collection(COLLECTION)
    print(f"✓ Qdrant '{COLLECTION}': {info.points_count} points live")


if __name__ == "__main__":
    main()
