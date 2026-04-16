"""Eval metrics: custom citation accuracy + RAGAS wrappers.

The custom citation_accuracy metric measures source attribution quality
(F1 over returned vs expected source IDs). RAGAS handles the four
generation/retrieval-quality metrics (faithfulness, answer_relevancy,
context_precision, context_recall) via an Ollama OpenAI-compatible backend.
"""

from __future__ import annotations


def citation_accuracy(
    returned_sources: list[str], expected_sources: list[str]
) -> dict[str, float]:
    """F1 over source IDs (file paths or chunk IDs).

    Returns precision, recall, f1. Empty cases return zeros so the metric is
    safe to aggregate across a dataset without special-casing.
    """
    returned_set = set(returned_sources)
    expected_set = set(expected_sources)

    tp = len(returned_set & expected_set)
    fp = len(returned_set - expected_set)
    fn = len(expected_set - returned_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}
