#!/usr/bin/env python
"""Run full A/B benchmark suite across retrieval configurations.

Usage:
    OLLAMA_HOST=http://172.19.64.1:11434 uv run python tests/eval/run_benchmarks.py

Evaluates 9 configurations (3 strategies x 3 top_k values) against the
30-query golden dataset. Writes JSON results to tests/eval/results/ and
PNG plots to docs/benchmarks/. No LLM judge needed -- all metrics are
retrieval-only.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi_rag_lab.eval.benchmarks import (
    RetrieverConfig,
    print_benchmark_summary,
    run_benchmark,
)
from fastapi_rag_lab.eval.plotting import save_all_plots


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    dataset_path = Path(__file__).parent / "golden_dataset.json"
    results_dir = Path(__file__).parent / "results"
    plots_dir = Path(__file__).parents[2] / "docs" / "benchmarks"

    configs = [
        RetrieverConfig(strategy=strategy, top_k=top_k)
        for strategy in ("dense", "hybrid", "hybrid_rerank")
        for top_k in (3, 5, 10)
    ]

    results = run_benchmark(configs, dataset_path, results_dir)
    print_benchmark_summary(results)

    plot_paths = save_all_plots(results, plots_dir)
    for p in plot_paths:
        logging.info("Saved plot: %s", p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
