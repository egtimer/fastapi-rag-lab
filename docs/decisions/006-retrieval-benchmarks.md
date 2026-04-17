# ADR 006: Retrieval-Only A/B Benchmarks

## Status
Accepted -- 2026-04-17

## Context

ADR 005 established a full evaluation pipeline: RAGAS metrics (faithfulness,
answer relevancy, context precision, context recall) plus a custom citation
accuracy metric, all driven by `gemma3:4b` as the LLM judge. That pipeline
works for a single configuration at a time, but it is too slow for comparative
benchmarks.

The problem: a single RAGAS eval run takes 15-30 minutes for 30 queries because
every query requires multiple LLM generation calls (faithfulness alone needs
claim extraction + verification). Benchmarking 9 configurations would take
2-4 hours on the local Ollama setup, and the results would be noisy because
`gemma3:4b` is not a particularly strong judge model.

What we actually want to know for the A/B comparison is simpler: *does hybrid
retrieval find more relevant documents than dense-only? Does the reranker
improve ranking quality? What is the latency cost?* These are retrieval
questions, not generation questions. They can be answered without an LLM.

## Decision

Benchmark retrieval quality across 9 configurations using retrieval-only
metrics that require no LLM inference:

**Metrics:**
- **Citation Accuracy F1** -- F1 over returned vs expected source files
  (existing metric from `eval/metrics.py`)
- **Recall@K** -- fraction of expected sources found in top-K retrieved
- **MRR (Mean Reciprocal Rank)** -- 1/rank of first correct source
- **nDCG@K** -- normalized discounted cumulative gain (binary relevance)
- **Latency** -- p50, p95, p99 in milliseconds per retrieval

**Configurations (3 x 3 = 9):**

| Dimension    | Values                               |
|--------------|--------------------------------------|
| Strategy     | dense, hybrid (RRF), hybrid + rerank |
| top_k        | 3, 5, 10                             |

The three strategies exercise progressively more of the retrieval pipeline:
dense uses only the Qdrant vector search, hybrid adds BM25 + RRF fusion,
hybrid+rerank adds the `bge-reranker-base` cross-encoder on top.

**Implementation:**
- `eval/retrieval_metrics.py` -- Recall@K, MRR, nDCG@K functions
- `eval/benchmarks.py` -- `RetrieverConfig` dataclass, `ConfiguredRetriever`
  adapter, `run_benchmark()` orchestrator
- `eval/plotting.py` -- four matplotlib charts saved as PNG
- `tests/eval/run_benchmarks.py` -- CLI entry point

Components are shared across configs: the BM25 index is built once (not per
config), and the cross-encoder model is loaded once. This keeps the total
benchmark time under 15 minutes.

## How to re-run

```bash
OLLAMA_HOST=http://172.19.64.1:11434 uv run python tests/eval/run_benchmarks.py
```

Results land in `tests/eval/results/benchmark_<timestamp>.json`.
Plots land in `docs/benchmarks/`.

## Consequences

- The full RAGAS eval pipeline (ADR 005) remains available for single-config
  deep evaluation. The benchmarks complement it for comparative analysis.
- Retrieval-only metrics cannot detect generation regressions. If we change
  the prompt template or switch LLMs, a RAGAS run is still needed.
- The golden dataset's `expected_sources` field does double duty: ground truth
  for both RAGAS context metrics and the retrieval-only metrics here.
- Adding new retrieval strategies (e.g., hypothetical document embeddings,
  query expansion) only requires a new `strategy` variant in `RetrieverConfig`.
