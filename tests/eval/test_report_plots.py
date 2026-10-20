"""Offline tests for the README plots drawn from committed reports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi_rag_lab.eval.report_plots import main

if TYPE_CHECKING:
    from pathlib import Path


def _stat(mean):
    return {"mean": mean, "ci_low": mean - 0.05, "ci_high": mean + 0.05}


def _write_run(root: Path, name: str, strategies: list[str]) -> Path:
    d = root / name
    d.mkdir()
    configs = [
        {
            "config_label": f"{s}_k5",
            "strategy": s,
            "top_k": 5,
            "n": 3,
            "recall_at_k": _stat(0.5),
            "mrr": _stat(0.4),
        }
        for s in strategies
    ]
    per_query = [
        {"config_label": f"{s}_k5", "query_id": f"q{i}", "latency_ms": 10.0 * (i + 1)}
        for s in strategies
        for i in range(3)
    ]
    (d / "analysis.json").write_text(json.dumps({"configs": configs}))
    (d / "benchmark.json").write_text(json.dumps({"per_query": per_query}))
    return d


def test_writes_both_plots(tmp_path: Path):
    dev = _write_run(tmp_path, "dev", ["sparse", "hybrid", "hybrid_rerank"])
    test = _write_run(tmp_path, "test", ["sparse", "hybrid", "hybrid_rerank"])
    out = tmp_path / "plots"
    argv = ["--run", f"dev={dev}", "--run", f"test={test}", "--out", str(out)]
    assert main(argv) == 0
    assert (out / "quality_ci.png").stat().st_size > 0
    assert (out / "latency.png").stat().st_size > 0


def test_refuses_a_top_k_missing_from_the_runs(tmp_path: Path):
    dev = _write_run(tmp_path, "dev", ["hybrid"])
    out = tmp_path / "plots"
    assert main(["--run", f"dev={dev}", "--top-k", "10", "--out", str(out)]) == 2
