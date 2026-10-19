"""Offline tests for the embedding runtime record and check."""

import httpx
import pytest

from fastapi_rag_lab.ingest import runtime_check
from fastapi_rag_lab.ingest.chunker import Child, Parent
from fastapi_rag_lab.ingest.embedder import embedding_runtime
from fastapi_rag_lab.ingest.runtime_check import check_runtime_matches_index
from fastapi_rag_lab.ingest.store import VECTOR_DIMENSION, upsert_children

HOST = "http://fake:11434"
RUNTIME = {
    "ollama_version": "0.35.1",
    "embedding_model": "nomic-embed-text",
    "embedding_model_digest": "abc",
}


def _fake_ollama(monkeypatch, *, version="0.35.1", models=None, fail=False):
    if models is None:
        models = [{"name": "nomic-embed-text:latest", "digest": "abc"}]

    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("down")
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": version})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": models})
        return httpx.Response(404)

    real = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)


@pytest.mark.parametrize("name", ["nomic-embed-text", "nomic-embed-text:latest"])
def test_embedding_runtime_parses(monkeypatch, name):
    _fake_ollama(
        monkeypatch,
        models=[{"name": "other", "digest": "x"}, {"name": name, "digest": "abc"}],
    )
    assert embedding_runtime(HOST) == RUNTIME


def test_embedding_runtime_model_missing(monkeypatch):
    _fake_ollama(monkeypatch, models=[{"name": "gemma3:4b", "digest": "x"}])
    with pytest.raises(RuntimeError, match="not found"):
        embedding_runtime(HOST)


def test_embedding_runtime_unreachable(monkeypatch):
    _fake_ollama(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="Cannot determine"):
        embedding_runtime(HOST)


def test_embedding_runtime_no_version(monkeypatch):
    _fake_ollama(monkeypatch, version="")
    with pytest.raises(RuntimeError, match="no version"):
        embedding_runtime(HOST)


class _Point:
    def __init__(self, payload):
        self.payload = payload


def _fake_qdrant(monkeypatch, points):
    class FakeClient:
        def __init__(self, url):
            pass

        def scroll(self, **kwargs):
            assert kwargs["limit"] == 1 and kwargs["with_payload"] is True
            return points, None

    monkeypatch.setattr(runtime_check, "QdrantClient", FakeClient)


def test_check_match(monkeypatch):
    _fake_ollama(monkeypatch)
    _fake_qdrant(monkeypatch, [_Point({**RUNTIME, "child_text": "x"})])
    r = check_runtime_matches_index("http://q", "c", HOST)
    assert r["match"] is True and r["index"] == r["query"] == RUNTIME


def test_check_mismatch(monkeypatch):
    _fake_ollama(monkeypatch)
    _fake_qdrant(monkeypatch, [_Point({**RUNTIME, "ollama_version": "0.20.7"})])
    r = check_runtime_matches_index("http://q", "c", HOST)
    assert r["match"] is False
    assert r["index"]["ollama_version"] == "0.20.7"
    assert r["query"]["ollama_version"] == "0.35.1"


@pytest.mark.parametrize("points", [[_Point({"child_text": "x"})], []])
def test_check_old_index(monkeypatch, points):
    _fake_ollama(monkeypatch)
    _fake_qdrant(monkeypatch, points)
    r = check_runtime_matches_index("http://q", "c", HOST)
    assert r["match"] is None and r["reason"] == "index has no runtime fields"


def test_store_writes_runtime_into_payloads(monkeypatch):
    captured = []

    class FakeClient:
        def __init__(self, url):
            pass

        def upsert(self, collection_name, points):
            captured.extend(points)

    monkeypatch.setattr("fastapi_rag_lab.ingest.store.QdrantClient", FakeClient)
    monkeypatch.setattr(
        "fastapi_rag_lab.ingest.store._ensure_collection", lambda c, n: None
    )
    parent = Parent(
        parent_id="p", text="t", source_file="a.md", heading_path=["A"], token_count=1
    )
    children = [
        Child(
            child_id=f"{i:016x}", parent_id="p", text="c", child_index=i, token_count=1
        )
        for i in range(2)
    ]
    vecs = [[0.0] * VECTOR_DIMENSION for _ in children]
    upsert_children([parent], children, vecs, qdrant_url="http://q", runtime=RUNTIME)
    assert len(captured) == 2
    for pt in captured:
        for k, v in RUNTIME.items():
            assert pt.payload[k] == v
