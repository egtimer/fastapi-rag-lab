"""Pydantic models for retrieval pipeline results."""

from __future__ import annotations

from pydantic import BaseModel


class RetrievalCandidate(BaseModel):
    """A single retrieval result before fusion."""

    chunk_id: str
    parent_id: str
    score: float
    text: str
    metadata: dict


class FusedResult(BaseModel):
    """Result after RRF fusion."""

    chunk_id: str
    parent_id: str
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    text: str
    metadata: dict


class RerankedResult(BaseModel):
    """Final reranked result with parent context."""

    chunk_id: str
    parent_id: str
    parent_text: str
    chunk_text: str
    rerank_score: float
    rrf_score: float
    metadata: dict
