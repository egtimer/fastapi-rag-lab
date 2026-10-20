"""README plots from committed benchmark reports, with bootstrap CIs.

Reads ``analysis.json`` (means and 95% CIs) and ``benchmark.json`` (per-query
latencies) of one or more runs and draws, at one top_k:

- ``quality_ci.png``: Recall@k and MRR per strategy, one bar per run, with CIs
- ``latency.png``: per-query latency per strategy, one box per run, log scale

    uv run python -m fastapi_rag_lab.eval.report_plots \\
        --run dev=reports/<dev-run> --run test=reports/<test-run> --out docs/benchmarks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fastapi_rag_lab.eval.plotting import STRATEGY_LABELS

STRATEGIES = ("sparse", "dense", "hybrid", "hybrid_rerank")
METRICS = (("recall_at_k", "Recall@{k}"), ("mrr", "MRR"))
RUN_COLOURS = ("#4c72b0", "#dd8452", "#55a868", "#c44e52")


def _label(strategy: str) -> str:
    return STRATEGY_LABELS[strategy].replace(" + ", "\n+ ")


def _load(run_dir: Path) -> tuple[dict, dict]:
    analysis = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
    return analysis, report


def _configs(analysis: dict, top_k: int) -> dict[str, dict]:
    return {c["strategy"]: c for c in analysis["configs"] if c["top_k"] == top_k}


def _strategies(runs: dict[str, tuple[dict, dict]], top_k: int) -> list[str]:
    present = set.intersection(*(set(_configs(a, top_k)) for a, _ in runs.values()))
    return [s for s in STRATEGIES if s in present]


def _n(analysis: dict, top_k: int) -> int:
    return next(iter(_configs(analysis, top_k).values()))["n"]


def plot_quality(
    runs: dict[str, tuple[dict, dict]], top_k: int, output_dir: Path
) -> Path:
    strategies = _strategies(runs, top_k)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12, 5), sharey=True)
    x = np.arange(len(strategies))
    width = 0.8 / len(runs)
    for ax, (metric, title) in zip(axes, METRICS, strict=True):
        for i, (name, (analysis, _)) in enumerate(runs.items()):
            configs = _configs(analysis, top_k)
            stats = [configs[s][metric] for s in strategies]
            means = np.array([st["mean"] for st in stats])
            err = np.array(
                [
                    [st["mean"] - st["ci_low"] for st in stats],
                    [st["ci_high"] - st["mean"] for st in stats],
                ]
            )
            offset = (i - (len(runs) - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=err,
                capsize=4,
                color=RUN_COLOURS[i % len(RUN_COLOURS)],
                label=f"{name} (n={_n(analysis, top_k)})",
            )
            for xi, m in zip(x + offset, means, strict=True):
                ax.text(xi, 0.005, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title.format(k=top_k))
        ax.set_xticks(x)
        ax.set_xticklabels([_label(s) for s in strategies], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean over questions, 95% bootstrap CI")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(runs)
    )
    fig.suptitle(f"Retrieval quality by strategy (top_k={top_k})")
    path = output_dir / "quality_ci.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_latency(
    runs: dict[str, tuple[dict, dict]], top_k: int, output_dir: Path
) -> Path:
    strategies = _strategies(runs, top_k)
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(runs)
    for i, (name, (_, report)) in enumerate(runs.items()):
        data = [
            [
                r["latency_ms"]
                for r in report["per_query"]
                if r["config_label"] == f"{s}_k{top_k}"
            ]
            for s in strategies
        ]
        positions = np.arange(len(strategies)) + (i - (len(runs) - 1) / 2) * width
        bp = ax.boxplot(
            data, positions=positions, widths=width * 0.9, patch_artist=True
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(RUN_COLOURS[i % len(RUN_COLOURS)])
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
        ax.plot([], [], "s", color=RUN_COLOURS[i % len(RUN_COLOURS)], label=name)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_xticklabels([_label(s) for s in strategies])
    ax.set_ylabel("Latency per query (ms, log scale)")
    ax.set_title(f"Retrieval latency by strategy (top_k={top_k}, CPU only)")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend()
    path = output_dir / "latency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _run_arg(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=_run_arg, action="append", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    runs = {name: _load(path) for name, path in args.run}
    if not _strategies(runs, args.top_k):
        print(f"error: no strategy at top_k={args.top_k} in all runs", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    for path in (
        plot_quality(runs, args.top_k, args.out),
        plot_latency(runs, args.top_k, args.out),
    ):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
