"""FastAPI application with lifespan management."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langfuse import Langfuse

from fastapi_rag_lab.api.routes.query import router as query_router
from fastapi_rag_lab.retrieval.hybrid import STRATEGIES, HybridRetriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup, clean up on shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    strategy = os.environ.get("RAG_RETRIEVAL_STRATEGY", "hybrid_rerank")
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
