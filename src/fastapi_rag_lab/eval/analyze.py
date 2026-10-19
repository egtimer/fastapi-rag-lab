"""Re-analyse a benchmark report with bootstrap confidence intervals.

Usage:
    python -m fastapi_rag_lab.eval.analyze <benchmark.json> [--out analysis.json]
        [--resamples 10000] [--seed 0]

Reads the JSON written by ``benchmarks.py``, recomputes per-config means from
the per-query rows (and checks them against the stored aggregates), adds 95%
CIs, and runs paired bootstraps between strategies at each top_k.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from fastapi_rag_lab.eval.stats import bootstrap_ci, paired_bootstrap

METRICS = ("recall_at_k", "mrr", "ndcg_at_k", "citation_f1")
STRATEGY_ORDER = ("sparse", "dense", "hybrid", "hybrid_rerank")
TOLERANCE = 1e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_report(
    report: dict[str, Any], *, n_resamples: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """Compute CIs and paired comparisons for a loaded benchmark report."""
    rows_by_label: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in report["per_query"]:
        rows_by_label[row["config_label"]][row["query_id"]] = row

    configs: list[dict[str, Any]] = []
    for agg in report["aggregates"]:
        label = agg["config_label"]
        rows = rows_by_label.get(label)
        if not rows:
            raise ValueError(f"no per-query rows for config {label}")
        ordered = [rows[qid] for qid in sorted(rows)]
        entry: dict[str, Any] = {
            "config_label": label,
            "strategy": agg["strategy"],
            "top_k": agg["top_k"],
            "n": len(ordered),
        }
        for metric in METRICS:
            mean, low, high = bootstrap_ci(
                [r[metric] for r in ordered], n_resamples=n_resamples, seed=seed
            )
            stored = agg[f"{metric}_mean"]
            if abs(mean - stored) > TOLERANCE:
                raise ValueError(
                    f"{label} {metric}: recomputed mean {mean!r} != "
                    f"stored aggregate {stored!r}"
                )
            entry[metric] = {"mean": mean, "ci_low": low, "ci_high": high}
        configs.append(entry)

    by_k: dict[int, dict[str, str]] = defaultdict(dict)
    for agg in report["aggregates"]:
        by_k[agg["top_k"]][agg["strategy"]] = agg["config_label"]

    comparisons: list[dict[str, Any]] = []
    for top_k in sorted(by_k):
        present = [s for s in STRATEGY_ORDER if s in by_k[top_k]]
        present += sorted(set(by_k[top_k]) - set(STRATEGY_ORDER))
        # (a, b) with b the "later" strategy: reports b - a.
        for a_name, b_name in combinations(present, 2):
            rows_a = rows_by_label[by_k[top_k][a_name]]
            rows_b = rows_by_label[by_k[top_k][b_name]]
            if set(rows_a) != set(rows_b):
                raise ValueError(
                    f"query_id sets differ for {a_name} vs {b_name} at k={top_k}"
                )
            qids = sorted(rows_a)
            for metric in METRICS:
                res = paired_bootstrap(
                    [rows_a[q][metric] for q in qids],
                    [rows_b[q][metric] for q in qids],
                    n_resamples=n_resamples,
                    seed=seed,
                )
                comparisons.append(
                    {
                        "top_k": top_k,
                        "a": a_name,
                        "b": b_name,
                        "metric": metric,
                        "diff": "b - a",
                        "mean_diff": res.mean_diff,
                        "ci_low": res.ci_low,
                        "ci_high": res.ci_high,
                        "frac_resamples_diff_le_0": res.frac_resamples_diff_le_0,
                        "n": res.n,
                        "wins_b": res.wins_b,
                        "wins_a": res.wins_a,
                        "ties": res.ties,
                    }
                )
    return {"configs": configs, "comparisons": comparisons}


def format_markdown(analysis: dict[str, Any]) -> str:
    """Render a compact markdown summary."""
    lines = ["## Per-config means (95% bootstrap CI)", ""]
    lines.append("| config | n | " + " | ".join(METRICS) + " |")
    lines.append("|---|---|" + "---|" * len(METRICS))
    for c in analysis["configs"]:
        cells = [
            f"{c[m]['mean']:.3f} [{c[m]['ci_low']:.3f}, {c[m]['ci_high']:.3f}]"
            for m in METRICS
        ]
        lines.append(f"| {c['config_label']} | {c['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences (b - a, 95% CI, wins b/a/ties)", ""]
    lines.append("| k | b - a | metric | diff | 95% CI | P(diff<=0) | b/a/tie |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in analysis["comparisons"]:
        lines.append(
            f"| {p['top_k']} | {p['b']} - {p['a']} | {p['metric']} | "
            f"{p['mean_diff']:+.3f} | [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}] | "
            f"{p['frac_resamples_diff_le_0']:.3f} | "
            f"{p['wins_b']}/{p['wins_a']}/{p['ties']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("benchmark", type=Path, help="benchmark JSON report")
    parser.add_argument("--out", type=Path, default=None, help="write full JSON here")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    report = json.loads(args.benchmark.read_text())
    analysis = analyze_report(report, n_resamples=args.resamples, seed=args.seed)
    print(format_markdown(analysis))

    if args.out is not None:
        command = "python -m fastapi_rag_lab.eval.analyze " + " ".join(
            sys.argv[1:] if argv is None else argv
        )
        out = {
            "input_path": str(args.benchmark),
            "input_sha256": _sha256(args.benchmark),
            "resamples": args.resamples,
            "seed": args.seed,
            "alpha": 0.05,
            "command": command,
            **analysis,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
