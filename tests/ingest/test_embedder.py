import httpx
import pytest

from fastapi_rag_lab.ingest.embedder import (
    BATCH_SIZE,
    EXPECTED_DIMENSION,
    embed_texts,
)


def _make_ollama_response(count: int, dim: int = EXPECTED_DIMENSION):
    return httpx.Response(
        200,
        json={"embeddings": [[0.1] * dim for _ in range(count)]},
        request=httpx.Request("POST", "http://fake:11434/api/embed"),
    )


def test_embed_batches_requests(monkeypatch):
    texts = [f"text {i}" for i in range(BATCH_SIZE + 3)]
    call_log = []

    def mock_post(self, url, *, json, **kwargs):
        call_log.append(len(json["input"]))
        return _make_ollama_response(len(json["input"]))

    monkeypatch.setattr(httpx.Client, "post", mock_post)
    monkeypatch.setenv("OLLAMA_HOST", "http://fake:11434")

    vectors = embed_texts(texts)

    assert len(vectors) == len(texts)
    assert call_log == [BATCH_SIZE, 3]


def test_embed_validates_dimension(monkeypatch):
    def mock_post(self, url, *, json, **kwargs):
        return _make_ollama_response(len(json["input"]), dim=512)

    monkeypatch.setattr(httpx.Client, "post", mock_post)
    monkeypatch.setenv("OLLAMA_HOST", "http://fake:11434")

    with pytest.raises(RuntimeError, match="failed after"):
        embed_texts(["hello"])


def test_embed_retries_on_failure(monkeypatch):
    attempt_count = 0

    def mock_post(self, url, *, json, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise httpx.ConnectError("connection refused")
        return _make_ollama_response(len(json["input"]))

    monkeypatch.setattr(httpx.Client, "post", mock_post)
    monkeypatch.setattr("fastapi_rag_lab.ingest.embedder.time.sleep", lambda _: None)
    monkeypatch.setenv("OLLAMA_HOST", "http://fake:11434")

    vectors = embed_texts(["hello"])
    assert len(vectors) == 1
    assert attempt_count == 3


def _ollama_reachable() -> bool:
    import os

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=5)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.ollama
@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not reachable")
def test_embed_real_ollama():
    texts = ["FastAPI is a modern web framework", "Dependency injection"]
    vectors = embed_texts(texts)

    assert len(vectors) == 2
    assert all(len(v) == EXPECTED_DIMENSION for v in vectors)
    assert all(isinstance(v[0], float) for v in vectors)
