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
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
RUNTIME_TIMEOUT = 10.0
RUNTIME_FIELDS = ("ollama_version", "embedding_model", "embedding_model_digest")


def default_ollama_host() -> str:
    return (os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")


def embedding_runtime(ollama_host: str | None = None) -> dict[str, str]:
    """Return the Ollama server version and embedding model digest.

    Embeddings depend on the server version (Ollama 0.30.0 changed how
    nomic-embed-text treats case), so both are recorded with the index.
    Raises RuntimeError if either cannot be determined.
    """
    host = (ollama_host or default_ollama_host()).rstrip("/")
    try:
        with httpx.Client(timeout=RUNTIME_TIMEOUT) as client:
            resp = client.get(f"{host}/api/version")
            resp.raise_for_status()
            version = resp.json().get("version")
            resp = client.get(f"{host}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        raise RuntimeError(
            f"Cannot determine the Ollama runtime at {host}: {exc}"
        ) from exc

    if not version:
        raise RuntimeError(f"Ollama at {host} returned no version from /api/version")

    names = {EMBEDDING_MODEL, f"{EMBEDDING_MODEL}:latest"}
    digest = None
    for m in models:
        if m.get("name") in names or m.get("model") in names:
            digest = m.get("digest")
            break
    if not digest:
        raise RuntimeError(
            f"Embedding model {EMBEDDING_MODEL!r} (or :latest) not found in "
            f"{host}/api/tags; pull it first"
        )
    return {
        "ollama_version": str(version),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_model_digest": str(digest),
    }


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
