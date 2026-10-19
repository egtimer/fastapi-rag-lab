"""Hybrid retrieval: dense + sparse + RRF, optionally reranked by a cross-encoder."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from fastapi_rag_lab.ingest.store import COLLECTION_NAME

from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .reranker import Reranker
from .sparse import SparseRetriever
from .types import RerankedResult

if TYPE_CHECKING:
    from .types import FusedResult

logger = logging.getLogger(__name__)

Strategy = Literal["hybrid", "hybrid_rerank"]
STRATEGIES: tuple[Strategy, ...] = ("hybrid", "hybrid_rerank")


class HybridRetriever:
    """Hybrid retrieval pipeline used by the query API.

    Pipeline:
        1. Dense + sparse retrieval (top-30 each)
        2. RRF fusion -> top-15 candidates
        3. ``hybrid_rerank``: cross-encoder rerank -> top-5 final results;
           ``hybrid``: the top-5 fused candidates in RRF order
        4. Results include parent text for context expansion

    Components can be injected (tests); otherwise they are built from the
    Qdrant collection, and the reranker model is loaded only for ``hybrid_rerank``.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        *,
        strategy: Strategy = "hybrid_rerank",
        dense: DenseRetriever | None = None,
        sparse: SparseRetriever | None = None,
        reranker: Reranker | None = None,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}, expected one of {STRATEGIES}"
            )
        self.strategy = strategy
        self.dense = dense if dense is not None else DenseRetriever(collection_name)
        self.sparse = sparse if sparse is not None else SparseRetriever(collection_name)
        self.reranker: Reranker | None = None
        if strategy == "hybrid_rerank":
            self.reranker = reranker if reranker is not None else Reranker()

    def retrieve(
        self,
        query: str,
        dense_k: int = 30,
        sparse_k: int = 30,
        fusion_k: int = 15,
        final_k: int = 5,
        *,
        rrf_k: int = 60,
    ) -> list[RerankedResult]:
        """Run the full hybrid retrieval pipeline."""
        dense_results = self.dense.retrieve(query, top_k=dense_k)
        sparse_results = self.sparse.retrieve(query, top_k=sparse_k)

        logger.info(
            "Dense: %d results, Sparse: %d results",
            len(dense_results),
            len(sparse_results),
        )

        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=rrf_k,
            top_k=fusion_k,
        )

        if self.reranker is None:
            return [_unranked(r) for r in fused[:final_k]]

        return self.reranker.rerank(
            query=query,
            candidates=fused,
            top_k=final_k,
        )


def _unranked(result: FusedResult) -> RerankedResult:
    """Wrap a fused candidate as a final result without a rerank score."""
    return RerankedResult(
        chunk_id=result.chunk_id,
        parent_id=result.parent_id,
        parent_text=result.metadata.get("parent_text", result.text),
        chunk_text=result.text,
        rerank_score=0.0,
        rrf_score=result.rrf_score,
        metadata=result.metadata,
    )
