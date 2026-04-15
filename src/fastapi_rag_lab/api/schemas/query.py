"""Pydantic models for the /query endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query from the user."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = True
    model: str | None = None


class Citation(BaseModel):
    """A single source citation returned alongside the answer."""

    source_id: str
    parent_id: str
    relevance_score: float
    excerpt: str = Field(max_length=200)
    source_file: str = ""
    heading_path: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Full response for non-streaming mode."""

    answer: str
    citations: list[Citation]
    latency_ms: int
    trace_id: str


class StreamChunk(BaseModel):
    """A single SSE event payload."""

    type: str  # "token" | "citations" | "done" | "error"
    content: str | list[Citation] | dict | None = None
