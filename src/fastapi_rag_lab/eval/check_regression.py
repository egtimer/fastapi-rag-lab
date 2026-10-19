"""Gate a benchmark run against a committed reference run.

Both runs must come from the same dataset (sha256 in provenance), contain the
chosen config, and cover the same query ids. For each metric the per-query
difference current - reference is bootstrapped over queries. A metric fails if
its mean drop exceeds ``--max-drop`` or if the whole 95% CI lies below zero.
Per-query change counts and the latency p50 ratio are reported, not gated.

Exit codes: 0 pass, 1 regression, 2 runs not comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from fastapi_rag_lab.eval.stats import paired_bootstrap

N_RESAMPLES = 10_000
SEED = 0
DEFAULT_METRICS = "recall_at_k,mrr"


class NotComparableError(Exception):
    """The two runs cannot be compared (exit code 2)."""


@dataclass(frozen=True)
class MetricCheck:
    metric: str
    reference: float
    current: float
    change: float
    ci_low: float
    ci_high: float
    limit: float
    passed: bool
    improved: int
    worsened: int


@dataclass(frozen=True)
class RegressionReport:
    config: str
    n_queries: int
    checks: list[MetricCheck]
    latency_p50_ratio: float | None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _load(run_dir: Path) -> dict:
    path = run_dir / "benchmark.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NotComparableError(f"cannot read {path}: {exc}") from exc


def _dataset_sha(report: dict, run_dir: Path) -> str:
    sha = ((report.get("provenance") or {}).get("dataset") or {}).get("sha256")
    if not sha:
        raise NotComparableError(f"{run_dir}: no dataset sha256 in provenance")
    return sha


def _rows(report: dict, config: str, run_dir: Path) -> dict[str, dict]:
    rows = {
        r["query_id"]: r
        for r in report.get("per_query", [])
        if r["config_label"] == config
    }
    if not rows:
        raise NotComparableError(f"config {config!r} not found in {run_dir}")
    return rows


def check(
    reference_dir: Path,
    current_dir: Path,
    *,
    config: str = "hybrid_k5",
    metrics: list[str] | None = None,
    max_drop: float = 0.05,
) -> RegressionReport:
    metrics = metrics or DEFAULT_METRICS.split(",")
    ref, cur = _load(reference_dir), _load(current_dir)
    ref_sha = _dataset_sha(ref, reference_dir)
    cur_sha = _dataset_sha(cur, current_dir)
    if ref_sha != cur_sha:
        raise NotComparableError(
            f"dataset sha256 differs: reference {ref_sha[:12]} "
            f"vs current {cur_sha[:12]}"
        )
    ref_rows = _rows(ref, config, reference_dir)
    cur_rows = _rows(cur, config, current_dir)
    if ref_rows.keys() != cur_rows.keys():
        only_ref = sorted(ref_rows.keys() - cur_rows.keys())[:5]
        only_cur = sorted(cur_rows.keys() - ref_rows.keys())[:5]
        raise NotComparableError(
            f"query_id sets differ (only in reference: {only_ref}, "
            f"only in current: {only_cur})"
        )
    ids = sorted(ref_rows)
    for m in metrics:
        if any(m not in ref_rows[i] or m not in cur_rows[i] for i in ids):
            raise NotComparableError(f"metric {m!r} missing from per-query rows")

    checks = []
    for m in metrics:
        a = [float(ref_rows[i][m]) for i in ids]
        b = [float(cur_rows[i][m]) for i in ids]
        res = paired_bootstrap(a, b, n_resamples=N_RESAMPLES, seed=SEED)
        failed = -res.mean_diff > max_drop or res.ci_high < 0
        checks.append(
            MetricCheck(
                metric=m,
                reference=sum(a) / len(a),
                current=sum(b) / len(b),
                change=res.mean_diff,
                ci_low=res.ci_low,
                ci_high=res.ci_high,
                limit=-max_drop,
                passed=not failed,
                improved=res.wins_b,
                worsened=res.wins_a,
            )
        )
    ref_lat = statistics.median(float(ref_rows[i]["latency_ms"]) for i in ids)
    cur_lat = statistics.median(float(cur_rows[i]["latency_ms"]) for i in ids)
    ratio = cur_lat / ref_lat if ref_lat > 0 else None
    return RegressionReport(config, len(ids), checks, ratio)


def format_markdown(report: RegressionReport, *, max_drop: float) -> str:
    lines = [
        f"### Retrieval regression check: `{report.config}` "
        f"({report.n_queries} queries)",
        "",
        "| Metric | Reference | Current | Change [95% CI] | Limit | Status |",
        "|---|---|---|---|---|---|",
    ]
    for c in report.checks:
        lines.append(
            f"| {c.metric} | {c.reference:.3f} | {c.current:.3f} "
            f"| {c.change:+.3f} [{c.ci_low:+.3f}, {c.ci_high:+.3f}] "
            f"| drop <= {max_drop:.3f}, CI upper >= 0 "
            f"| {'pass' if c.passed else 'FAIL'} |"
        )
    lines.append("")
    lines.append("Information only (not gated):")
    for c in report.checks:
        lines.append(
            f"- {c.metric}: {c.improved} queries improved, {c.worsened} worsened"
        )
    if report.latency_p50_ratio is not None:
        lines.append(
            f"- latency p50, current / reference: {report.latency_p50_ratio:.2f}x"
        )
    lines.append("")
    lines.append(f"Result: {'pass' if report.passed else 'fail'}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--current", type=Path, required=True)
    p.add_argument("--config", default="hybrid_k5")
    p.add_argument("--metrics", default=DEFAULT_METRICS, help="comma-separated")
    p.add_argument("--max-drop", type=float, default=0.05)
    args = p.parse_args(argv)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    try:
        report = check(
            args.reference,
            args.current,
            config=args.config,
            metrics=metrics,
            max_drop=args.max_drop,
        )
    except NotComparableError as exc:
        print(f"error: runs not comparable: {exc}", file=sys.stderr)
        return 2
    md = format_markdown(report, max_drop=args.max_drop)
    print(md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
