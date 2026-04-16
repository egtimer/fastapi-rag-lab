"""Evaluation runner: loads the golden dataset, drives retrieval + generation,
computes RAGAS + citation_accuracy, writes a JSON report and aggregates."""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from pathlib import Path

from fastapi_rag_lab.api.llm.ollama_client import OllamaClient
from fastapi_rag_lab.api.prompt import build_prompt
from fastapi_rag_lab.eval.dataset import GoldenEntry, load_golden_dataset
from fastapi_rag_lab.eval.metrics import (
    RagasBackend,
    build_ollama_backend,
    citation_accuracy,
    run_ragas_metrics,
)
from fastapi_rag_lab.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "context_precision": 0.70,
    "context_recall": 0.65,
    "citation_f1": 0.60,
}


@dataclass
class PerQueryResult:
    id: str
    query: str
    category: str
    retrieved_source_files: list[str]
    expected_sources: list[str]
    generated_answer: str
    reference_answer: str
    citation_precision: float
    citation_recall: float
    citation_f1: float
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None
    latency_ms: int
    error: str | None = None


@dataclass
class EvalResults:
    started_at: str
    finished_at: str
    dataset_path: str
    entry_count: int
    llm_model: str
    embed_model: str
    aggregate: dict[str, float | int | None] = field(default_factory=dict)
    threshold_status: dict[str, bool] = field(default_factory=dict)
    per_query: list[PerQueryResult] = field(default_factory=list)


def run_eval(
    dataset_path: Path,
    output_dir: Path,
    *,
    retriever: HybridRetriever | None = None,
    backend: RagasBackend | None = None,
    llm: OllamaClient | None = None,
    final_k: int = 5,
) -> EvalResults:
    """Run the full evaluation suite end-to-end and write a JSON report."""
    dataset = load_golden_dataset(dataset_path)
    retriever = retriever or HybridRetriever()
    backend = backend or build_ollama_backend()
    llm = llm or OllamaClient()

    started_at = datetime.now(UTC).isoformat()
    per_query: list[PerQueryResult] = []

    for entry in tqdm(dataset.entries, desc="eval", unit="q"):
        per_query.append(
            _evaluate_one(
                entry=entry,
                retriever=retriever,
                backend=backend,
                llm=llm,
                final_k=final_k,
            )
        )

    finished_at = datetime.now(UTC).isoformat()
    aggregate = _aggregate(per_query)
    threshold_status = _check_thresholds(aggregate)

    results = EvalResults(
        started_at=started_at,
        finished_at=finished_at,
        dataset_path=str(dataset_path),
        entry_count=len(dataset.entries),
        llm_model=backend.llm_model,
        embed_model=backend.embed_model,
        aggregate=aggregate,
        threshold_status=threshold_status,
        per_query=per_query,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"eval_{stamp}.json"
    out_path.write_text(_serialise(results), encoding="utf-8")
    logger.info("Wrote %s", out_path)

    return results


def _evaluate_one(
    *,
    entry: GoldenEntry,
    retriever: HybridRetriever,
    backend: RagasBackend,
    llm: OllamaClient,
    final_k: int,
) -> PerQueryResult:
    t0 = time.monotonic()

    try:
        retrieved = retriever.retrieve(entry.query, final_k=final_k)
    except Exception as exc:
        return _failed_result(entry, error=f"retrieval: {exc}", t0=t0)

    retrieved_source_files = [
        r.metadata.get("source_file", "") for r in retrieved if r.metadata
    ]
    retrieved_contexts = [r.parent_text for r in retrieved if r.parent_text]
    citation = citation_accuracy(
        returned_sources=retrieved_source_files,
        expected_sources=entry.expected_sources,
    )

    context_blocks = [(i + 1, ctx) for i, ctx in enumerate(retrieved_contexts)]
    prompt = build_prompt(entry.query, context_blocks)

    try:
        generated_answer = asyncio.run(llm.generate(prompt))
    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return PerQueryResult(
            id=entry.id,
            query=entry.query,
            category=entry.category,
            retrieved_source_files=retrieved_source_files,
            expected_sources=entry.expected_sources,
            generated_answer="",
            reference_answer=entry.reference_answer,
            citation_precision=citation["precision"],
            citation_recall=citation["recall"],
            citation_f1=citation["f1"],
            faithfulness=None,
            answer_relevancy=None,
            context_precision=None,
            context_recall=None,
            latency_ms=latency_ms,
            error=f"generation: {exc}",
        )

    ragas = run_ragas_metrics(
        query=entry.query,
        retrieved_contexts=retrieved_contexts,
        generated_answer=generated_answer,
        reference_answer=entry.reference_answer,
        backend=backend,
    )

    latency_ms = int((time.monotonic() - t0) * 1000)
    return PerQueryResult(
        id=entry.id,
        query=entry.query,
        category=entry.category,
        retrieved_source_files=retrieved_source_files,
        expected_sources=entry.expected_sources,
        generated_answer=generated_answer,
        reference_answer=entry.reference_answer,
        citation_precision=citation["precision"],
        citation_recall=citation["recall"],
        citation_f1=citation["f1"],
        faithfulness=ragas["faithfulness"],
        answer_relevancy=ragas["answer_relevancy"],
        context_precision=ragas["context_precision"],
        context_recall=ragas["context_recall"],
        latency_ms=latency_ms,
        error=None,
    )


def _failed_result(entry: GoldenEntry, *, error: str, t0: float) -> PerQueryResult:
    return PerQueryResult(
        id=entry.id,
        query=entry.query,
        category=entry.category,
        retrieved_source_files=[],
        expected_sources=entry.expected_sources,
        generated_answer="",
        reference_answer=entry.reference_answer,
        citation_precision=0.0,
        citation_recall=0.0,
        citation_f1=0.0,
        faithfulness=None,
        answer_relevancy=None,
        context_precision=None,
        context_recall=None,
        latency_ms=int((time.monotonic() - t0) * 1000),
        error=error,
    )


def _aggregate(rows: list[PerQueryResult]) -> dict[str, float | int | None]:
    def _mean(values: list[float | None]) -> float | None:
        clean = [v for v in values if v is not None]
        return statistics.fmean(clean) if clean else None

    def _coverage(values: list[float | None]) -> int:
        return sum(1 for v in values if v is not None)

    aggregate: dict[str, float | int | None] = {
        "citation_precision_mean": _mean([r.citation_precision for r in rows]),
        "citation_recall_mean": _mean([r.citation_recall for r in rows]),
        "citation_f1_mean": _mean([r.citation_f1 for r in rows]),
        "faithfulness_mean": _mean([r.faithfulness for r in rows]),
        "answer_relevancy_mean": _mean([r.answer_relevancy for r in rows]),
        "context_precision_mean": _mean([r.context_precision for r in rows]),
        "context_recall_mean": _mean([r.context_recall for r in rows]),
        "faithfulness_n": _coverage([r.faithfulness for r in rows]),
        "answer_relevancy_n": _coverage([r.answer_relevancy for r in rows]),
        "context_precision_n": _coverage([r.context_precision for r in rows]),
        "context_recall_n": _coverage([r.context_recall for r in rows]),
        "latency_ms_p50": _percentile([r.latency_ms for r in rows], 50),
        "latency_ms_p95": _percentile([r.latency_ms for r in rows], 95),
        "error_count": sum(1 for r in rows if r.error),
    }
    return aggregate


def _percentile(values: list[int], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return float(ordered[f])
    return float(ordered[f] + (ordered[c] - ordered[f]) * (k - f))


def _check_thresholds(aggregate: dict[str, float | int | None]) -> dict[str, bool]:
    """A metric below threshold (or with no successful samples) is failing."""
    status: dict[str, bool] = {}
    mapping = {
        "faithfulness": "faithfulness_mean",
        "answer_relevancy": "answer_relevancy_mean",
        "context_precision": "context_precision_mean",
        "context_recall": "context_recall_mean",
        "citation_f1": "citation_f1_mean",
    }
    for metric, threshold in THRESHOLDS.items():
        observed = aggregate.get(mapping[metric])
        is_numeric = isinstance(observed, (int, float)) and observed is not None
        status[metric] = is_numeric and observed >= threshold
    return status


def _serialise(results: EvalResults) -> str:
    payload = asdict(results)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def print_summary(results: EvalResults) -> None:
    """Render a one-page summary of the run for stdout consumption."""
    a = results.aggregate
    print("\n=== Eval summary ===")
    print(f"Dataset:        {results.dataset_path}")
    print(f"Entries:        {results.entry_count}")
    print(f"LLM:            {results.llm_model}")
    print(f"Embeddings:     {results.embed_model}")
    print(f"Errors:         {a.get('error_count')}")
    print(
        f"Latency (ms):   p50={a.get('latency_ms_p50')}  "
        f"p95={a.get('latency_ms_p95')}"
    )
    print()
    print(f"{'metric':<22} {'mean':>8}  {'n':>4}  {'thr':>5}  status")
    print("-" * 56)

    n_total = results.entry_count
    rows = [
        ("citation_precision", a.get("citation_precision_mean"), n_total, None, None),
        ("citation_recall", a.get("citation_recall_mean"), n_total, None, None),
        (
            "citation_f1",
            a.get("citation_f1_mean"),
            n_total,
            THRESHOLDS["citation_f1"],
            results.threshold_status.get("citation_f1"),
        ),
        (
            "faithfulness",
            a.get("faithfulness_mean"),
            a.get("faithfulness_n"),
            THRESHOLDS["faithfulness"],
            results.threshold_status.get("faithfulness"),
        ),
        (
            "answer_relevancy",
            a.get("answer_relevancy_mean"),
            a.get("answer_relevancy_n"),
            THRESHOLDS["answer_relevancy"],
            results.threshold_status.get("answer_relevancy"),
        ),
        (
            "context_precision",
            a.get("context_precision_mean"),
            a.get("context_precision_n"),
            THRESHOLDS["context_precision"],
            results.threshold_status.get("context_precision"),
        ),
        (
            "context_recall",
            a.get("context_recall_mean"),
            a.get("context_recall_n"),
            THRESHOLDS["context_recall"],
            results.threshold_status.get("context_recall"),
        ),
    ]
    for name, mean, n, thr, status in rows:
        mean_s = f"{mean:.3f}" if isinstance(mean, (int, float)) else "  n/a"
        thr_s = f"{thr:.2f}" if thr is not None else "  -- "
        if status is True:
            status_s = "PASS"
        elif status is False:
            status_s = "FAIL"
        else:
            status_s = " -- "
        print(f"{name:<22} {mean_s:>8}  {n:>4}  {thr_s:>5}  {status_s}")
    print()
