"""Retrieval-only IR metrics: Recall@K, MRR, nDCG@K.

These metrics evaluate retrieval quality without requiring an LLM judge,
making them fast enough to run across many configurations in A/B benchmarks.
All functions accept lists of source identifiers (file paths) and compare
retrieved vs expected sets.
"""

from __future__ import annotations

import math


def recall_at_k(
    retrieved_sources: list[str], expected_sources: list[str], k: int
) -> float:
    """Fraction of expected_sources found in the top-K retrieved."""
    if not expected_sources:
        return 1.0
    top_k = set(retrieved_sources[:k])
    hits = len(top_k & set(expected_sources))
    return hits / len(expected_sources)


def mrr(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first correct source in retrieved list."""
    expected_set = set(expected_sources)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_sources: list[str], expected_sources: list[str], k: int
) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Binary relevance: 1 if source is in expected_sources, 0 otherwise.
    """
    if not expected_sources:
        return 1.0
    expected_set = set(expected_sources)
    top_k = retrieved_sources[:k]

    # DCG: sum of rel_i / log2(i+1) for i in 1..k
    dcg = sum(
        1.0 / math.log2(i + 2)  # i+2 because enumerate starts at 0
        for i, source in enumerate(top_k)
        if source in expected_set
    )

    # Ideal DCG: all relevant docs at the top positions
    n_relevant = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
