"""A/B benchmark runner: evaluates retrieval quality across configurations.

Runs retrieval-only metrics (no LLM judge) across a cross-product of
retrieval strategies and top_k values. Each configuration is evaluated
against every query in the golden dataset, producing per-query and
aggregate results.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from tqdm import tqdm

from fastapi_rag_lab.eval.dataset import load_golden_dataset
from fastapi_rag_lab.eval.metrics import citation_accuracy
from fastapi_rag_lab.eval.retrieval_metrics import mrr, ndcg_at_k, recall_at_k
from fastapi_rag_lab.retrieval.dense import DenseRetriever
from fastapi_rag_lab.retrieval.fusion import reciprocal_rank_fusion
from fastapi_rag_lab.retrieval.reranker import Reranker
from fastapi_rag_lab.retrieval.sparse import SparseRetriever
from fastapi_rag_lab.retrieval.types import RerankedResult

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi_rag_lab.eval.dataset import GoldenEntry

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    strategy: Literal["dense", "hybrid", "hybrid_rerank"]
    top_k: int
    rrf_k: int = 60
    rerank_top_n: int = 15

    @property
    def label(self) -> str:
        return f"{self.strategy}_k{self.top_k}"


@dataclass
class PerQueryBenchmark:
    config_label: str
    query_id: str
    query: str
    category: str
    retrieved_sources: list[str]
    expected_sources: list[str]
    citation_f1: float
    citation_precision: float
    citation_recall: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_ms: float


@dataclass
class ConfigAggregate:
    config_label: str
    strategy: str
    top_k: int
    citation_f1_mean: float
    recall_at_k_mean: float
    mrr_mean: float
    ndcg_at_k_mean: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class BenchmarkResults:
    started_at: str
    finished_at: str
    dataset_path: str
    entry_count: int
    configs: list[dict] = field(default_factory=list)
    aggregates: list[ConfigAggregate] = field(default_factory=list)
    per_query: list[PerQueryBenchmark] = field(default_factory=list)


class ConfiguredRetriever:
    """Retriever assembled from components according to a config."""

    def __init__(
        self,
        config: RetrieverConfig,
        dense: DenseRetriever,
        sparse: SparseRetriever | None,
        reranker: Reranker | None,
    ):
        self.config = config
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker

    def retrieve(self, query: str) -> list[RerankedResult]:
        k = self.config.top_k
        cfg = self.config

        if cfg.strategy == "dense":
            dense_results = self.dense.retrieve(query, top_k=k)
            return [
                RerankedResult(
                    chunk_id=r.chunk_id,
                    parent_id=r.parent_id,
                    parent_text=r.metadata.get("parent_text", r.text),
                    chunk_text=r.text,
                    rerank_score=0.0,
                    rrf_score=r.score,
                    metadata=r.metadata,
                )
                for r in dense_results
            ]

        # hybrid and hybrid_rerank both need sparse + fusion
        dense_results = self.dense.retrieve(query, top_k=30)
        sparse_results = self.sparse.retrieve(query, top_k=30)
        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=cfg.rrf_k,
            top_k=cfg.rerank_top_n,
        )

        if cfg.strategy == "hybrid":
            return [
                RerankedResult(
                    chunk_id=r.chunk_id,
                    parent_id=r.parent_id,
                    parent_text=r.metadata.get("parent_text", r.text),
                    chunk_text=r.text,
                    rerank_score=0.0,
                    rrf_score=r.rrf_score,
                    metadata=r.metadata,
                )
                for r in fused[:k]
            ]

        # hybrid_rerank
        return self.reranker.rerank(query=query, candidates=fused, top_k=k)


def build_retriever(
    config: RetrieverConfig,
    dense: DenseRetriever,
    sparse: SparseRetriever | None = None,
    reranker: Reranker | None = None,
) -> ConfiguredRetriever:
    """Factory returning a ConfiguredRetriever for the given strategy."""
    if config.strategy in ("hybrid", "hybrid_rerank") and sparse is None:
        raise ValueError(f"{config.strategy} requires a SparseRetriever")
    if config.strategy == "hybrid_rerank" and reranker is None:
        raise ValueError("hybrid_rerank requires a Reranker")
    return ConfiguredRetriever(config, dense, sparse, reranker)


def run_benchmark(
    configs: list[RetrieverConfig],
    dataset_path: Path,
    output_dir: Path,
) -> BenchmarkResults:
    """Run retrieval-only benchmarks across all configs and queries."""
    dataset = load_golden_dataset(dataset_path)
    started_at = datetime.now(UTC).isoformat()

    # Build shared components once (BM25 index + reranker model are expensive)
    logger.info("Initializing retrievers...")
    dense = DenseRetriever()
    needs_sparse = any(c.strategy in ("hybrid", "hybrid_rerank") for c in configs)
    needs_reranker = any(c.strategy == "hybrid_rerank" for c in configs)
    sparse = SparseRetriever() if needs_sparse else None
    reranker = Reranker() if needs_reranker else None

    per_query: list[PerQueryBenchmark] = []

    for config in configs:
        retriever = build_retriever(config, dense, sparse, reranker)
        logger.info("Benchmarking %s ...", config.label)

        for entry in tqdm(
            dataset.entries, desc=config.label, unit="q", leave=False
        ):
            result = _benchmark_one(retriever, entry, config)
            per_query.append(result)

    finished_at = datetime.now(UTC).isoformat()
    aggregates = _compute_aggregates(per_query, configs)

    results = BenchmarkResults(
        started_at=started_at,
        finished_at=finished_at,
        dataset_path=str(dataset_path),
        entry_count=len(dataset.entries),
        configs=[asdict(c) for c in configs],
        aggregates=aggregates,
        per_query=per_query,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"benchmark_{stamp}.json"
    out_path.write_text(
        json.dumps(asdict(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote %s", out_path)

    return results


def _benchmark_one(
    retriever: ConfiguredRetriever,
    entry: GoldenEntry,
    config: RetrieverConfig,
) -> PerQueryBenchmark:
    t0 = time.monotonic()
    try:
        retrieved = retriever.retrieve(entry.query)
    except Exception as exc:
        logger.warning("Retrieval failed for %s on %s: %s", config.label, entry.id, exc)
        elapsed_ms = (time.monotonic() - t0) * 1000
        return PerQueryBenchmark(
            config_label=config.label,
            query_id=entry.id,
            query=entry.query,
            category=entry.category,
            retrieved_sources=[],
            expected_sources=entry.expected_sources,
            citation_f1=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            latency_ms=elapsed_ms,
        )
    elapsed_ms = (time.monotonic() - t0) * 1000

    # Deduplicate sources preserving rank order -- multiple child chunks from
    # the same source file should not inflate ranking metrics (nDCG, MRR).
    seen: set[str] = set()
    sources: list[str] = []
    for r in retrieved:
        src = r.metadata.get("source_file", "")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    k = config.top_k
    cit = citation_accuracy(sources, entry.expected_sources)

    return PerQueryBenchmark(
        config_label=config.label,
        query_id=entry.id,
        query=entry.query,
        category=entry.category,
        retrieved_sources=sources,
        expected_sources=entry.expected_sources,
        citation_f1=cit["f1"],
        citation_precision=cit["precision"],
        citation_recall=cit["recall"],
        recall_at_k=recall_at_k(sources, entry.expected_sources, k),
        mrr=mrr(sources, entry.expected_sources),
        ndcg_at_k=ndcg_at_k(sources, entry.expected_sources, k),
        latency_ms=elapsed_ms,
    )


def _compute_aggregates(
    per_query: list[PerQueryBenchmark],
    configs: list[RetrieverConfig],
) -> list[ConfigAggregate]:
    aggregates = []
    for config in configs:
        rows = [r for r in per_query if r.config_label == config.label]
        if not rows:
            continue

        latencies = sorted(r.latency_ms for r in rows)

        # Per-category breakdown
        categories: dict[str, list[PerQueryBenchmark]] = {}
        for r in rows:
            categories.setdefault(r.category, []).append(r)

        per_cat = {}
        for cat, cat_rows in categories.items():
            per_cat[cat] = {
                "recall_at_k": statistics.fmean(r.recall_at_k for r in cat_rows),
                "mrr": statistics.fmean(r.mrr for r in cat_rows),
                "citation_f1": statistics.fmean(r.citation_f1 for r in cat_rows),
                "ndcg_at_k": statistics.fmean(r.ndcg_at_k for r in cat_rows),
            }

        aggregates.append(
            ConfigAggregate(
                config_label=config.label,
                strategy=config.strategy,
                top_k=config.top_k,
                citation_f1_mean=statistics.fmean(r.citation_f1 for r in rows),
                recall_at_k_mean=statistics.fmean(r.recall_at_k for r in rows),
                mrr_mean=statistics.fmean(r.mrr for r in rows),
                ndcg_at_k_mean=statistics.fmean(r.ndcg_at_k for r in rows),
                latency_p50_ms=_percentile(latencies, 50),
                latency_p95_ms=_percentile(latencies, 95),
                latency_p99_ms=_percentile(latencies, 99),
                per_category=per_cat,
            )
        )
    return aggregates


def _percentile(sorted_values: list[float], pct: int) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def print_benchmark_summary(results: BenchmarkResults) -> None:
    """Print a summary table of benchmark results."""
    print("\n=== Benchmark Summary ===")
    print(f"Dataset: {results.dataset_path} ({results.entry_count} queries)")
    print(f"Configs: {len(results.aggregates)}")
    print(f"Duration: {results.started_at} -> {results.finished_at}")
    print()

    header = (
        f"{'config':<22} {'Cit.F1':>7} {'Rec@K':>7} {'MRR':>7} "
        f"{'nDCG':>7} {'p50ms':>7} {'p95ms':>7}"
    )
    print(header)
    print("-" * len(header))

    for agg in results.aggregates:
        print(
            f"{agg.config_label:<22} "
            f"{agg.citation_f1_mean:>7.3f} "
            f"{agg.recall_at_k_mean:>7.3f} "
            f"{agg.mrr_mean:>7.3f} "
            f"{agg.ndcg_at_k_mean:>7.3f} "
            f"{agg.latency_p50_ms:>7.1f} "
            f"{agg.latency_p95_ms:>7.1f}"
        )
    print()
