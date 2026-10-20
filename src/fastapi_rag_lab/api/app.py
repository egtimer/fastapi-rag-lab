"""FastAPI application with lifespan management."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langfuse import Langfuse

from fastapi_rag_lab.api.routes.query import router as query_router
from fastapi_rag_lab.ingest.embedder import default_ollama_host
from fastapi_rag_lab.ingest.runtime_check import (
    check_runtime_matches_index,
    format_runtime,
)
from fastapi_rag_lab.ingest.store import COLLECTION_NAME
from fastapi_rag_lab.retrieval.hybrid import STRATEGIES, HybridRetriever

logger = logging.getLogger(__name__)


def verify_embedding_runtime() -> None:
    """Refuse to start if the Ollama runtime differs from the index's."""
    check = check_runtime_matches_index(
        os.environ.get("QDRANT_URL", "http://localhost:6333"),
        COLLECTION_NAME,
        default_ollama_host(),
    )
    if check["match"] is None:
        logger.warning("Embedding runtime not verified: %s", check.get("reason"))
    elif check["match"] is False:
        msg = (
            f"Embedding runtime mismatch: index built with "
            f"{format_runtime(check['index'])}; querying with "
            f"{format_runtime(check['query'])}. Query vectors would not match the "
            f"index. Use the pinned Ollama or re-ingest."
        )
        if os.environ.get("RAG_ALLOW_RUNTIME_MISMATCH") == "1":
            logger.warning("%s (RAG_ALLOW_RUNTIME_MISMATCH=1, continuing)", msg)
        else:
            raise RuntimeError(
                msg + " Set RAG_ALLOW_RUNTIME_MISMATCH=1 to start anyway."
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup, clean up on shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    verify_embedding_runtime()

    strategy = os.environ.get("RAG_RETRIEVAL_STRATEGY", "hybrid")
    if strategy not in STRATEGIES:
        raise ValueError(
            f"RAG_RETRIEVAL_STRATEGY must be one of {STRATEGIES}, got {strategy!r}"
        )
    logger.info("Initializing HybridRetriever (strategy=%s)...", strategy)
    app.state.retriever = HybridRetriever(strategy=strategy)
    app.state.langfuse = Langfuse()
    logger.info("Ready to serve queries")

    yield

    logger.info("Shutting down, flushing Langfuse...")
    app.state.langfuse.flush()


app = FastAPI(
    title="fastapi-rag-lab",
    description="RAG query API over FastAPI documentation",
    lifespan=lifespan,
)

app.include_router(query_router)


@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}
