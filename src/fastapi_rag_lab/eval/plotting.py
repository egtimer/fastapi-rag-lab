"""Benchmark visualization: generates PNG plots from BenchmarkResults.

Uses the Agg backend so it works headless in WSL / Docker / CI.
All figures saved at 300 DPI for README embedding.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi_rag_lab.eval.benchmarks import BenchmarkResults, ConfigAggregate

# Consistent colours per strategy across all plots
STRATEGY_COLOURS = {
    "dense": "#4c72b0",
    "hybrid": "#55a868",
    "hybrid_rerank": "#c44e52",
}
STRATEGY_LABELS = {
    "dense": "Dense",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + Reranker",
}


def save_all_plots(results: BenchmarkResults, output_dir: Path) -> list[Path]:
    """Generate all benchmark plots and return paths to saved PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_quality_by_strategy(results, output_dir),
        plot_recall_at_k(results, output_dir),
        plot_latency_distribution(results, output_dir),
        plot_per_category(results, output_dir),
    ]
    plt.close("all")
    return paths


def plot_quality_by_strategy(results: BenchmarkResults, output_dir: Path) -> Path:
    """Bar chart: retrieval quality metrics grouped by strategy (top_k=5)."""
    aggregates = [a for a in results.aggregates if a.top_k == 5]
    if not aggregates:
        aggregates = results.aggregates[:3]

    metrics = ["citation_f1_mean", "recall_at_k_mean", "ndcg_at_k_mean", "mrr_mean"]
    metric_labels = ["Citation F1", "Recall@K", "nDCG@K", "MRR"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_labels))
    n_strategies = len(aggregates)
    width = 0.8 / n_strategies

    for i, agg in enumerate(aggregates):
        values = [getattr(agg, m) for m in metrics]
        offset = (i - (n_strategies - 1) / 2) * width
        colour = STRATEGY_COLOURS.get(agg.strategy, "#999999")
        label = STRATEGY_LABELS.get(agg.strategy, agg.strategy)
        bars = ax.bar(x + offset, values, width, label=label, color=colour)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Score")
    ax.set_title("Retrieval Quality by Strategy (top_k=5)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = output_dir / "quality_by_strategy.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def plot_recall_at_k(results: BenchmarkResults, output_dir: Path) -> Path:
    """Line chart: Recall@K vs K value, one line per strategy."""
    strategies = ["dense", "hybrid", "hybrid_rerank"]
    k_values = sorted({a.top_k for a in results.aggregates})

    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy in strategies:
        recall_values = []
        valid_k = []
        for k in k_values:
            agg = _find_aggregate(results.aggregates, strategy, k)
            if agg:
                recall_values.append(agg.recall_at_k_mean)
                valid_k.append(k)
        if valid_k:
            colour = STRATEGY_COLOURS.get(strategy, "#999999")
            label = STRATEGY_LABELS.get(strategy, strategy)
            ax.plot(
                valid_k, recall_values, "o-", color=colour, label=label,
                linewidth=2, markersize=8,
            )
            for k_val, recall_val in zip(valid_k, recall_values):
                ax.annotate(
                    f"{recall_val:.2f}",
                    (k_val, recall_val),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                )

    ax.set_xlabel("K (number of retrieved documents)")
    ax.set_ylabel("Recall@K")
    ax.set_title("Recall@K by Retrieval Strategy")
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)

    path = output_dir / "recall_at_k.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def plot_latency_distribution(results: BenchmarkResults, output_dir: Path) -> Path:
    """Box plot: latency distribution by strategy (all top_k values combined)."""
    strategies = ["dense", "hybrid", "hybrid_rerank"]

    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    colours = []

    for strategy in strategies:
        latencies = [
            r.latency_ms
            for r in results.per_query
            if r.config_label.startswith(strategy)
        ]
        if latencies:
            data.append(latencies)
            labels.append(STRATEGY_LABELS.get(strategy, strategy))
            colours.append(STRATEGY_COLOURS.get(strategy, "#999999"))

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Retrieval Latency Distribution by Strategy")
    ax.grid(axis="y", alpha=0.3)

    path = output_dir / "latency_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def plot_per_category(results: BenchmarkResults, output_dir: Path) -> Path:
    """Grouped bar: Recall@5 by category for each strategy."""
    aggregates = [a for a in results.aggregates if a.top_k == 5]
    if not aggregates:
        aggregates = results.aggregates[:3]

    all_categories = sorted(
        {cat for a in aggregates for cat in a.per_category}
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(all_categories))
    n_strategies = len(aggregates)
    width = 0.8 / max(n_strategies, 1)

    for i, agg in enumerate(aggregates):
        values = [
            agg.per_category.get(cat, {}).get("recall_at_k", 0.0)
            for cat in all_categories
        ]
        offset = (i - (n_strategies - 1) / 2) * width
        colour = STRATEGY_COLOURS.get(agg.strategy, "#999999")
        label = STRATEGY_LABELS.get(agg.strategy, agg.strategy)
        bars = ax.bar(x + offset, values, width, label=label, color=colour)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("Recall@5")
    ax.set_title("Per-Category Recall@5 by Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in all_categories])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = output_dir / "per_category_recall.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def _find_aggregate(
    aggregates: list[ConfigAggregate], strategy: str, top_k: int
) -> ConfigAggregate | None:
    for a in aggregates:
        if a.strategy == strategy and a.top_k == top_k:
            return a
    return None
