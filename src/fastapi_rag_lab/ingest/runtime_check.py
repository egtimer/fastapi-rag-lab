"""Compare the embedding runtime recorded in an index with the current one."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from fastapi_rag_lab.ingest.embedder import RUNTIME_FIELDS, embedding_runtime

NO_RUNTIME_REASON = "index has no runtime fields"


def check_runtime_matches_index(
    qdrant_url: str, collection: str, ollama_host: str
) -> dict[str, Any]:
    """Return ``{"index": ..., "query": ..., "match": bool | None}``.

    ``match`` is None (with a ``reason``) when the index predates runtime fields.
    Raises RuntimeError if the current Ollama runtime cannot be determined.
    """
    client = QdrantClient(url=qdrant_url)
    points, _ = client.scroll(
        collection_name=collection, limit=1, with_payload=True, with_vectors=False
    )
    payload = (points[0].payload or {}) if points else {}
    index = {f: payload.get(f) for f in RUNTIME_FIELDS}
    query = embedding_runtime(ollama_host)
    if any(v is None for v in index.values()):
        return {
            "index": index if any(v is not None for v in index.values()) else None,
            "query": query,
            "match": None,
            "reason": NO_RUNTIME_REASON,
        }
    return {"index": index, "query": query, "match": index == query}


def format_runtime(runtime: dict[str, Any] | None) -> str:
    if not runtime:
        return "unknown"
    return (
        f"ollama {runtime.get('ollama_version')}, {runtime.get('embedding_model')} "
        f"digest {runtime.get('embedding_model_digest')}"
    )
