"""Eval metrics: custom citation accuracy + RAGAS wrappers.

The custom ``citation_accuracy`` metric measures source attribution quality
(F1 over returned vs expected source IDs). RAGAS handles the four
generation/retrieval-quality metrics (faithfulness, answer_relevancy,
context_precision, context_recall) via an Ollama OpenAI-compatible backend.

The RAGAS wrapper computes each metric independently and traps exceptions
per-metric: a small LLM occasionally returns invalid structured output and
we don't want one bad query to abort the whole run. Failed metrics surface
as ``None`` in the returned dict so the runner can count them separately.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_LLM_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"


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


# --- RAGAS wiring ---------------------------------------------------------


@dataclass
class RagasBackend:
    """Container for the LLM + embeddings instances RAGAS needs."""

    llm: object
    embeddings: object
    llm_model: str
    embed_model: str


def build_ollama_backend(
    llm_model: str | None = None,
    embed_model: str | None = None,
    base_url: str | None = None,
) -> RagasBackend:
    """Wire RAGAS to talk to Ollama via its OpenAI-compatible /v1 API."""
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory

    base = base_url or _ollama_base_url()
    llm_name = llm_model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_LLM_MODEL)
    embed_name = embed_model or os.environ.get(
        "OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL
    )

    # api_key is required by the openai client but Ollama ignores its value.
    # Both the LLM and the embeddings call drive aembed_text/agenerate from
    # inside RAGAS' .ascore(), so both need the AsyncOpenAI client.
    async_client = AsyncOpenAI(base_url=f"{base}/v1", api_key="ollama")
    llm = llm_factory(llm_name, client=async_client)
    embeddings = OpenAIEmbeddings(client=async_client, model=embed_name)

    return RagasBackend(
        llm=llm,
        embeddings=embeddings,
        llm_model=llm_name,
        embed_model=embed_name,
    )


def run_ragas_metrics(
    *,
    query: str,
    retrieved_contexts: list[str],
    generated_answer: str,
    reference_answer: str,
    backend: RagasBackend,
) -> dict[str, float | None]:
    """Compute the four RAGAS metrics for one (query, answer) pair.

    Returns a dict with keys ``faithfulness``, ``answer_relevancy``,
    ``context_precision``, ``context_recall``. Each value is a float on
    [0, 1] or ``None`` if that specific metric raised (e.g. the LLM
    returned malformed structured output for it).
    """
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    faithfulness = Faithfulness(llm=backend.llm)
    answer_relevancy = AnswerRelevancy(llm=backend.llm, embeddings=backend.embeddings)
    context_precision = ContextPrecision(llm=backend.llm)
    context_recall = ContextRecall(llm=backend.llm)

    results: dict[str, float | None] = {
        "faithfulness": _safe_score(
            faithfulness.ascore(
                user_input=query,
                response=generated_answer,
                retrieved_contexts=retrieved_contexts,
            ),
            metric_name="faithfulness",
        ),
        "answer_relevancy": _safe_score(
            answer_relevancy.ascore(
                user_input=query,
                response=generated_answer,
            ),
            metric_name="answer_relevancy",
        ),
        "context_precision": _safe_score(
            context_precision.ascore(
                user_input=query,
                reference=reference_answer,
                retrieved_contexts=retrieved_contexts,
            ),
            metric_name="context_precision",
        ),
        "context_recall": _safe_score(
            context_recall.ascore(
                user_input=query,
                retrieved_contexts=retrieved_contexts,
                reference=reference_answer,
            ),
            metric_name="context_recall",
        ),
    }
    return results


def _safe_score(coro, *, metric_name: str) -> float | None:
    """Run an awaitable RAGAS metric and trap any exception."""
    try:
        result = asyncio.run(coro)
    except Exception as exc:
        logger.warning("RAGAS metric %s failed: %s", metric_name, exc)
        return None

    value = getattr(result, "value", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(
            "RAGAS metric %s returned non-numeric value: %r", metric_name, value
        )
        return None


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
