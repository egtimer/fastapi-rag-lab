import httpx
import pytest

from fastapi_rag_lab.retrieval.fusion import reciprocal_rank_fusion
from fastapi_rag_lab.retrieval.types import RetrievalCandidate


def _qdrant_reachable() -> bool:
    try:
        resp = httpx.get("http://localhost:6333/collections", timeout=3)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _ollama_reachable() -> bool:
    import os

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=3)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


# --- Unit tests (no infra needed) ---


def test_rrf_fusion_combines_rankings():
    """Candidates appearing in both lists get higher RRF scores."""
    dense = [
        RetrievalCandidate(
            chunk_id="a", parent_id="p1", score=0.9, text="...", metadata={}
        ),
        RetrievalCandidate(
            chunk_id="b", parent_id="p2", score=0.5, text="...", metadata={}
        ),
    ]
    sparse = [
        RetrievalCandidate(
            chunk_id="b", parent_id="p2", score=10.0, text="...", metadata={}
        ),
        RetrievalCandidate(
            chunk_id="a", parent_id="p1", score=5.0, text="...", metadata={}
        ),
    ]
    fused = reciprocal_rank_fusion(dense, sparse, top_k=2)
    assert len(fused) == 2
    assert all(f.rrf_score > 0 for f in fused)
    assert all(f.dense_rank is not None and f.sparse_rank is not None for f in fused)


def test_rrf_fusion_single_retriever_only():
    """Candidate appearing in only one list still gets a score."""
    dense = [
        RetrievalCandidate(
            chunk_id="a", parent_id="p1", score=0.9, text="...", metadata={}
        ),
    ]
    sparse = [
        RetrievalCandidate(
            chunk_id="b", parent_id="p2", score=10.0, text="...", metadata={}
        ),
    ]
    fused = reciprocal_rank_fusion(dense, sparse, top_k=2)
    assert len(fused) == 2
    # a has dense_rank but no sparse_rank, b has sparse_rank but no dense_rank
    a = next(f for f in fused if f.chunk_id == "a")
    b = next(f for f in fused if f.chunk_id == "b")
    assert a.dense_rank == 1 and a.sparse_rank is None
    assert b.sparse_rank == 1 and b.dense_rank is None


def test_rrf_empty_inputs():
    fused = reciprocal_rank_fusion([], [], top_k=5)
    assert fused == []


# --- Integration tests (need Qdrant + Ollama + indexed data) ---


@pytest.mark.qdrant
@pytest.mark.ollama
@pytest.mark.skipif(
    not _qdrant_reachable() or not _ollama_reachable(),
    reason="Qdrant or Ollama not reachable",
)
class TestDenseRetriever:
    def test_returns_results(self):
        from fastapi_rag_lab.retrieval.dense import DenseRetriever

        dense = DenseRetriever()
        results = dense.retrieve("How do I handle exceptions in FastAPI?", top_k=10)
        assert len(results) > 0
        assert all(r.score > 0 for r in results)


@pytest.mark.qdrant
@pytest.mark.skipif(not _qdrant_reachable(), reason="Qdrant not reachable")
class TestSparseRetriever:
    def test_finds_exact_keywords(self):
        from fastapi_rag_lab.retrieval.sparse import SparseRetriever

        sparse = SparseRetriever()
        results = sparse.retrieve("HTTPException", top_k=10)
        assert len(results) > 0
        assert any("HTTPException" in r.text for r in results)


@pytest.mark.qdrant
@pytest.mark.ollama
@pytest.mark.integration
@pytest.mark.skipif(
    not _qdrant_reachable() or not _ollama_reachable(),
    reason="Qdrant or Ollama not reachable",
)
class TestHybridRetriever:
    @pytest.fixture(scope="class")
    def hybrid(self):
        from fastapi_rag_lab.retrieval.hybrid import HybridRetriever

        return HybridRetriever()

    def test_returns_reranked_results(self, hybrid):
        results = hybrid.retrieve("How to handle exceptions in FastAPI?", final_k=5)
        assert len(results) == 5
        assert all(r.rerank_score is not None for r in results)
        assert all(r.parent_text != "" for r in results)
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_keyword_query_benefits_from_sparse(self, hybrid):
        """Sparse retrieval should surface exact keyword matches."""
        results = hybrid.retrieve("BackgroundTasks", final_k=5)
        assert len(results) > 0
        assert any(
            "BackgroundTasks" in r.chunk_text or "BackgroundTasks" in r.parent_text
            for r in results[:3]
        )

    def test_empty_query_handled_gracefully(self, hybrid):
        results = hybrid.retrieve("", final_k=5)
        assert isinstance(results, list)
