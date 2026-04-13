"""Reciprocal Rank Fusion for combining dense and sparse retrieval results."""

from __future__ import annotations

from .types import FusedResult, RetrievalCandidate


def reciprocal_rank_fusion(
    dense_results: list[RetrievalCandidate],
    sparse_results: list[RetrievalCandidate],
    k: int = 60,
    top_k: int = 15,
) -> list[FusedResult]:
    """Fuse two ranked lists via RRF.

    For each chunk: RRF score = sum over retrievers of 1 / (k + rank).
    k=60 from the original RRF paper (Cormack et al., 2009) dampens the
    impact of high ranks so low-ranked results still contribute.
    """
    rrf_scores: dict[str, dict] = {}

    for rank, candidate in enumerate(dense_results, start=1):
        cid = candidate.chunk_id
        if cid not in rrf_scores:
            rrf_scores[cid] = {
                "candidate": candidate,
                "score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
            }
        rrf_scores[cid]["score"] += 1.0 / (k + rank)
        rrf_scores[cid]["dense_rank"] = rank

    for rank, candidate in enumerate(sparse_results, start=1):
        cid = candidate.chunk_id
        if cid not in rrf_scores:
            rrf_scores[cid] = {
                "candidate": candidate,
                "score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
            }
        rrf_scores[cid]["score"] += 1.0 / (k + rank)
        rrf_scores[cid]["sparse_rank"] = rank

    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True,
    )[:top_k]

    return [
        FusedResult(
            chunk_id=r["candidate"].chunk_id,
            parent_id=r["candidate"].parent_id,
            rrf_score=r["score"],
            dense_rank=r["dense_rank"],
            sparse_rank=r["sparse_rank"],
            text=r["candidate"].text,
            metadata=r["candidate"].metadata,
        )
        for r in sorted_results
    ]
