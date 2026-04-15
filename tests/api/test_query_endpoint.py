"""Integration tests for POST /query. Require Ollama + Qdrant running."""

from __future__ import annotations

import json
import os

import httpx
import pytest

from fastapi_rag_lab.api.schemas.query import Citation


def _qdrant_reachable() -> bool:
    try:
        resp = httpx.get("http://localhost:6333/collections", timeout=3)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _ollama_reachable() -> bool:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=3)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


skip_unless_infra = pytest.mark.skipif(
    not _qdrant_reachable() or not _ollama_reachable(),
    reason="Qdrant or Ollama not reachable",
)


_app = None


def _get_app():
    """Lazy-init the app with shared state across tests."""
    global _app  # noqa: PLW0603
    if _app is None:
        from langfuse import Langfuse

        from fastapi_rag_lab.api.app import app
        from fastapi_rag_lab.retrieval.hybrid import HybridRetriever

        app.state.retriever = HybridRetriever()
        app.state.langfuse = Langfuse()
        _app = app
    return _app


@pytest.fixture
async def client():
    """Async test client, new per test to match event loop."""
    app = _get_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        yield ac


@skip_unless_infra
@pytest.mark.ollama
@pytest.mark.qdrant
@pytest.mark.integration
class TestQueryEndpoint:
    async def test_query_endpoint_returns_200(self, client):
        resp = await client.post(
            "/query",
            json={
                "query": "How do I handle exceptions in FastAPI?",
                "top_k": 3,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert len(body["answer"]) > 0
        assert "citations" in body
        assert "latency_ms" in body
        assert "trace_id" in body

    async def test_query_endpoint_streams_tokens(self, client):
        resp = await client.post(
            "/query",
            json={
                "query": "What is dependency injection in FastAPI?",
                "top_k": 3,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

        events = _parse_sse_events(resp.text)
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) > 0

    async def test_query_endpoint_includes_citations(self, client):
        resp = await client.post(
            "/query",
            json={
                "query": "How does FastAPI handle path parameters?",
                "top_k": 3,
                "stream": True,
            },
        )
        events = _parse_sse_events(resp.text)

        citation_events = [e for e in events if e["event"] == "citations"]
        assert len(citation_events) == 1
        payload = json.loads(citation_events[0]["data"])
        citations = payload["content"]
        assert len(citations) == 3

    async def test_query_endpoint_handles_empty_query(self, client):
        resp = await client.post(
            "/query",
            json={"query": "", "top_k": 3},
        )
        assert resp.status_code == 422

    async def test_query_endpoint_handles_llm_timeout(self, client):
        """Query with an unreachable model name triggers generation error."""
        resp = await client.post(
            "/query",
            json={
                "query": "test",
                "top_k": 3,
                "stream": True,
                "model": "nonexistent-model-xyz",
            },
        )
        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        event_types = [e["event"] for e in events]
        assert "error" in event_types or "token" in event_types

    async def test_citation_format_is_valid(self, client):
        resp = await client.post(
            "/query",
            json={
                "query": "How do I add middleware in FastAPI?",
                "top_k": 3,
                "stream": False,
            },
        )
        body = resp.json()
        for c in body["citations"]:
            citation = Citation(**c)
            assert citation.source_id
            assert citation.parent_id
            assert citation.relevance_score > 0
            assert len(citation.excerpt) <= 200

    async def test_langfuse_trace_created(self, client):
        resp = await client.post(
            "/query",
            json={
                "query": "What are response models in FastAPI?",
                "top_k": 3,
                "stream": False,
            },
        )
        body = resp.json()
        assert body["trace_id"]
        assert len(body["trace_id"]) > 0

    async def test_done_event_has_latency(self, client):
        resp = await client.post(
            "/query",
            json={
                "query": "How do I use BackgroundTasks?",
                "top_k": 3,
                "stream": True,
            },
        )
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1
        payload = json.loads(done_events[0]["data"])
        assert payload["content"]["latency_ms"] > 0
        assert payload["content"]["trace_id"]


def _parse_sse_events(text: str) -> list[dict]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data = None

    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            current_data = line[6:]
        elif line == "" and current_event is not None:
            events.append({"event": current_event, "data": current_data})
            current_event = None
            current_data = None

    return events
