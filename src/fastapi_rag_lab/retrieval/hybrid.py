"""Hybrid retrieval pipeline: dense + sparse + RRF + cross-encoder rerank."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi_rag_lab.ingest.store import COLLECTION_NAME

from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .reranker import Reranker
from .sparse import SparseRetriever

if TYPE_CHECKING:
    from .types import RerankedResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Production hybrid retrieval pipeline.

    Pipeline:
        1. Dense + sparse retrieval (top-30 each)
        2. RRF fusion -> top-15 candidates
        3. Cross-encoder rerank -> top-5 final results
        4. Results include parent text for context expansion
    """

    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.dense = DenseRetriever(collection_name)
        self.sparse = SparseRetriever(collection_name)
        self.reranker = Reranker()

    def retrieve(
        self,
        query: str,
        dense_k: int = 30,
        sparse_k: int = 30,
        fusion_k: int = 15,
        final_k: int = 5,
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
            top_k=fusion_k,
        )

        reranked = self.reranker.rerank(
            query=query,
            candidates=fused,
            top_k=final_k,
        )

        return reranked
