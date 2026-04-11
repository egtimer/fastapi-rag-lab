"""Embed child chunks via Ollama's nomic-embed-text model."""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"
EXPECTED_DIMENSION = 768
BATCH_SIZE = 16
REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 3


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed texts via Ollama and return parallel list of vectors."""
    ollama_host = os.environ.get("OLLAMA_HOST", "")
    if not ollama_host:
        ollama_host = "http://localhost:11434"
        logger.warning("OLLAMA_HOST not set, falling back to %s", ollama_host)

    embeddings: list[list[float]] = []

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[batch_start : batch_start + BATCH_SIZE]
            batch_vectors = _embed_batch(client, ollama_host, batch)
            embeddings.extend(batch_vectors)

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: sent {len(texts)} texts, "
            f"got {len(embeddings)} vectors"
        )

    return embeddings


def _embed_batch(
    client: httpx.Client,
    ollama_host: str,
    texts: list[str],
) -> list[list[float]]:
    url = f"{ollama_host}/api/embed"
    payload = {"model": EMBEDDING_MODEL, "input": texts}

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
            vectors = body["embeddings"]

            for i, vec in enumerate(vectors):
                if len(vec) != EXPECTED_DIMENSION:
                    raise ValueError(
                        f"Expected {EXPECTED_DIMENSION}-dim embedding, "
                        f"got {len(vec)} for text at index {i}"
                    )

            return vectors
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                wait = 2**attempt
                logger.warning(
                    "Embed batch failed (attempt %d/%d): %s. Retrying in %ds",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} attempts") from last_error
