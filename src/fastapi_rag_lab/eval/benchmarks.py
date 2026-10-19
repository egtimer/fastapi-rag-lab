"""A/B benchmark runner: evaluates retrieval quality across configurations.

Runs retrieval-only metrics (no LLM judge) across a cross-product of
retrieval strategies and top_k values. Each configuration is evaluated
against every query in the golden dataset, producing per-query and
aggregate results.
"""

from __future__ import annotations

import hashlib
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
from fastapi_rag_lab.eval.provenance import collect_provenance
from fastapi_rag_lab.eval.retrieval_metrics import mrr, ndcg_at_k, recall_at_k
from fastapi_rag_lab.retrieval.dense import DenseRetriever
from fastapi_rag_lab.retrieval.hybrid import HybridRetriever
from fastapi_rag_lab.retrieval.reranker import Reranker
from fastapi_rag_lab.retrieval.sparse import SparseRetriever
from fastapi_rag_lab.retrieval.types import RerankedResult

DENSE_K = 30
SPARSE_K = 30

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi_rag_lab.eval.dataset import GoldenEntry

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    strategy: Literal["sparse", "dense", "hybrid", "hybrid_rerank"]
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
    provenance: dict = field(default_factory=dict)


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
        # hybrid strategies run the exact class the API runs
        self.hybrid: HybridRetriever | None = None
        if config.strategy in ("hybrid", "hybrid_rerank"):
            self.hybrid = HybridRetriever(
                strategy=config.strategy,
                dense=dense,
                sparse=sparse,
                reranker=reranker,
            )

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

        if cfg.strategy == "sparse":
            sparse_results = self.sparse.retrieve(query, top_k=k)
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
                for r in sparse_results
            ]

        assert self.hybrid is not None
        return self.hybrid.retrieve(
            query,
            dense_k=DENSE_K,
            sparse_k=SPARSE_K,
            fusion_k=cfg.rerank_top_n,
            final_k=k,
            rrf_k=cfg.rrf_k,
        )


def build_retriever(
    config: RetrieverConfig,
    dense: DenseRetriever,
    sparse: SparseRetriever | None = None,
    reranker: Reranker | None = None,
) -> ConfiguredRetriever:
    """Factory returning a ConfiguredRetriever for the given strategy."""
    if config.strategy in ("sparse", "hybrid", "hybrid_rerank") and sparse is None:
        raise ValueError(f"{config.strategy} requires a SparseRetriever")
    if config.strategy == "hybrid_rerank" and reranker is None:
        raise ValueError("hybrid_rerank requires a Reranker")
    return ConfiguredRetriever(config, dense, sparse, reranker)


def run_benchmark(
    configs: list[RetrieverConfig],
    dataset_path: Path,
    output_path: Path | None = None,
    *,
    warmup: int = 0,
    argv: list[str] | None = None,
    checkpoint_path: Path | None = None,
    runtime_check: dict | None = None,
) -> BenchmarkResults:
    """Run retrieval-only benchmarks across all configs and queries.

    ``warmup`` unmeasured queries are run before timing each config.
    If ``output_path`` is given, the results (with provenance) are written there.
    If ``checkpoint_path`` is given, each finished config is appended to it as one
    JSON line, and configs already in it (same dataset sha256) are not run again,
    so an interrupted run can be resumed. Resumed configs are listed in
    ``provenance["resumed_configs"]``;
    ``runtime_check`` (see ``check_runtime_matches_index``) is recorded under
    ``provenance["embedding_runtime"]``; their latencies come from the earlier process.
    """
    dataset = load_golden_dataset(dataset_path)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    started_at = datetime.now(UTC).isoformat()
    done = _load_checkpoint(checkpoint_path, dataset_sha256, dataset.entries)
    pending = [c for c in configs if c.label not in done]
    if done:
        logger.info("Resuming: %d configs already done: %s", len(done), sorted(done))

    # Build shared components once (BM25 index + reranker model are expensive)
    logger.info("Initializing retrievers...")
    dense = DenseRetriever()
    needs_sparse = any(
        c.strategy in ("sparse", "hybrid", "hybrid_rerank") for c in pending
    )
    needs_reranker = any(c.strategy == "hybrid_rerank" for c in pending)
    sparse = SparseRetriever() if needs_sparse else None
    reranker = Reranker() if needs_reranker else None

    per_query_by_label: dict[str, list[PerQueryBenchmark]] = dict(done)

    for config in pending:
        retriever = build_retriever(config, dense, sparse, reranker)
        logger.info("Benchmarking %s ...", config.label)

        for entry in dataset.entries[:warmup]:
            try:
                retriever.retrieve(entry.query)
            except Exception as exc:
                logger.warning("Warmup failed for %s: %s", config.label, exc)

        rows = [
            _benchmark_one(retriever, entry, config)
            for entry in tqdm(dataset.entries, desc=config.label, unit="q", leave=False)
        ]
        per_query_by_label[config.label] = rows
        if checkpoint_path is not None:
            _append_checkpoint(checkpoint_path, dataset_sha256, config, rows)

    per_query = [row for c in configs for row in per_query_by_label[c.label]]
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
        provenance=collect_provenance(
            dataset_path=dataset_path,
            entry_count=len(dataset.entries),
            argv=argv,
            started_at=started_at,
            finished_at=finished_at,
        ),
    )
    results.provenance["embedding_runtime"] = runtime_check
    if done:
        results.provenance["resumed_configs"] = sorted(done)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(results), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote %s", output_path)

    return results


def _load_checkpoint(
    path: Path | None, dataset_sha256: str, entries: list[GoldenEntry]
) -> dict[str, list[PerQueryBenchmark]]:
    """Return the finished configs stored in a checkpoint file, by label."""
    if path is None or not path.exists():
        return {}
    expected_ids = [e.id for e in entries]
    done: dict[str, list[PerQueryBenchmark]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["dataset_sha256"] != dataset_sha256:
            raise ValueError(f"{path}: checkpoint was made on a different dataset")
        rows = [PerQueryBenchmark(**r) for r in record["rows"]]
        if [r.query_id for r in rows] != expected_ids:
            raise ValueError(f"{path}: {record['config_label']} rows do not match")
        done[record["config_label"]] = rows
    return done


def _append_checkpoint(
    path: Path,
    dataset_sha256: str,
    config: RetrieverConfig,
    rows: list[PerQueryBenchmark],
) -> None:
    """Append one finished config to the checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "config_label": config.label,
        "config": asdict(config),
        "dataset_sha256": dataset_sha256,
        "finished_at": datetime.now(UTC).isoformat(),
        "rows": [asdict(r) for r in rows],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
