"""Offline tests for the citation relevance score."""

from fastapi_rag_lab.api.routes.query import _build_citations
from fastapi_rag_lab.retrieval.hybrid import HybridRetriever
from fastapi_rag_lab.retrieval.types import RerankedResult


def _result(rerank_score, rrf_score):
    return RerankedResult(
        chunk_id="c1",
        parent_id="p1",
        parent_text="parent",
        chunk_text="chunk",
        rerank_score=rerank_score,
        rrf_score=rrf_score,
        metadata={"source_file": "tutorial/a.md"},
    )


def test_reranked_results_report_the_cross_encoder_score():
    (c,) = _build_citations([_result(0.91234, 0.03)], reranked=True)
    assert c.relevance_score == 0.9123


def test_fused_results_report_the_rrf_score():
    (c,) = _build_citations([_result(0.0, 0.032787)], reranked=False)
    assert c.relevance_score == 0.0328


def test_default_strategy_is_hybrid():
    r = HybridRetriever(dense=object(), sparse=object())
    assert r.strategy == "hybrid"
    assert r.reranker is None
