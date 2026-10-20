"""Cross-encoder reranking with bge-reranker-base."""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from .types import FusedResult, RerankedResult

logger = logging.getLogger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-base"
# Hugging Face commit of the model weights; a moving "main" could change the scores
RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"


class Reranker:
    """Cross-encoder reranker using BAAI/bge-reranker-base.

    Cross-encoders process [query, passage] pairs together with full
    attention, producing more accurate relevance scores than bi-encoders.
    The tradeoff is ~50ms per pair, so this only runs on the fused
    candidate set (top-15), not the full corpus.
    """

    def __init__(
        self, model_name: str = RERANKER_MODEL, revision: str = RERANKER_REVISION
    ):
        self.model = CrossEncoder(model_name, revision=revision, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[FusedResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """Score each candidate against the query and return top-K."""
        if not candidates:
            return []

        pairs = [[query, c.text] for c in candidates]
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True
        )

        results = []
        for candidate, rerank_score in scored[:top_k]:
            # parent_text is stored in the child payload metadata
            parent_text = candidate.metadata.get("parent_text", candidate.text)
            results.append(
                RerankedResult(
                    chunk_id=candidate.chunk_id,
                    parent_id=candidate.parent_id,
                    parent_text=parent_text,
                    chunk_text=candidate.text,
                    rerank_score=float(rerank_score),
                    rrf_score=candidate.rrf_score,
                    metadata=candidate.metadata,
                )
            )

        return results
