"""HybridRetriever strategies with injected fake components (no Qdrant, no models)."""

from __future__ import annotations

import pytest

from fastapi_rag_lab.retrieval.hybrid import HybridRetriever
from fastapi_rag_lab.retrieval.types import RerankedResult, RetrievalCandidate


def _candidate(i: int, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=f"c{i}",
        parent_id=f"p{i}",
        score=score,
        text=f"child {i}",
        metadata={"parent_text": f"parent {i}", "source": f"doc{i}.md"},
    )


class FakeRetriever:
    def __init__(self, order: list[int]):
        self.order = order
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 30) -> list[RetrievalCandidate]:
        self.calls.append((query, top_k))
        return [
            _candidate(i, 1.0 / (rank + 1)) for rank, i in enumerate(self.order[:top_k])
        ]


class FakeReranker:
    """Reverses the fused order, so its effect is visible."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, candidates, top_k=5) -> list[RerankedResult]:
        self.calls += 1
        return [
            RerankedResult(
                chunk_id=c.chunk_id,
                parent_id=c.parent_id,
                parent_text=c.metadata["parent_text"],
                chunk_text=c.text,
                rerank_score=float(n),
                rrf_score=c.rrf_score,
                metadata=c.metadata,
            )
            for n, c in enumerate(reversed(candidates[:top_k]))
        ]


def _retriever(strategy: str, reranker: FakeReranker | None = None) -> HybridRetriever:
    return HybridRetriever(
        strategy=strategy,
        dense=FakeRetriever([1, 2, 3, 4, 5, 6]),
        sparse=FakeRetriever([1, 3, 2, 6, 5, 4]),
        reranker=reranker,
    )


def test_hybrid_returns_fused_order_without_reranking():
    reranker = FakeReranker()
    retriever = _retriever("hybrid", reranker)

    results = retriever.retrieve("q", final_k=3)

    assert retriever.reranker is None
    assert reranker.calls == 0
    assert [r.chunk_id for r in results] == ["c1", "c2", "c3"]
    assert all(r.rerank_score == 0.0 for r in results)
    assert results[0].parent_text == "parent 1"
    assert results[0].rrf_score > results[1].rrf_score


def test_hybrid_rerank_uses_the_reranker():
    reranker = FakeReranker()
    retriever = _retriever("hybrid_rerank", reranker)

    results = retriever.retrieve("q", final_k=3)

    assert reranker.calls == 1
    assert [r.chunk_id for r in results] == ["c3", "c2", "c1"]


def test_both_strategies_retrieve_the_same_candidates():
    plain = _retriever("hybrid")
    reranked = _retriever("hybrid_rerank", FakeReranker())

    plain.retrieve("q", dense_k=30, sparse_k=30, fusion_k=15, final_k=5)
    reranked.retrieve("q", dense_k=30, sparse_k=30, fusion_k=15, final_k=5)

    assert plain.dense.calls == reranked.dense.calls == [("q", 30)]
    assert plain.sparse.calls == reranked.sparse.calls == [("q", 30)]


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        _retriever("dense")
