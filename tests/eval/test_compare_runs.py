"""Offline tests for the query-by-query run comparison."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi_rag_lab.eval.compare_runs import compare, main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _write(root: Path, name: str, rows: list[tuple[str, str, list[str], float]]):
    d = root / name
    d.mkdir()
    per_query = [
        {
            "config_label": c,
            "query_id": q,
            "retrieved_sources": s,
            "recall_at_k": r,
        }
        for c, q, s, r in rows
    ]
    (d / "benchmark.json").write_text(json.dumps({"per_query": per_query}))
    return d


def test_counts_identical_ordered_lists(tmp_path: Path):
    a = _write(
        tmp_path,
        "a",
        [("hybrid_k5", "q1", ["x", "y"], 1.0), ("hybrid_k5", "q2", ["x", "y"], 0.5)],
    )
    b = _write(
        tmp_path,
        "b",
        [("hybrid_k5", "q1", ["x", "y"], 1.0), ("hybrid_k5", "q2", ["y", "x"], 0.0)],
    )
    (row,) = compare(a, b)
    assert (row.config, row.n, row.identical) == ("hybrid_k5", 2, 1)
    assert (row.recall_a, row.recall_b) == (0.75, 0.5)


def test_only_shared_configs_are_compared(tmp_path: Path):
    a = _write(tmp_path, "a", [("dense_k5", "q1", ["x"], 1.0)])
    b = _write(
        tmp_path,
        "b",
        [("dense_k5", "q1", ["x"], 1.0), ("sparse_k5", "q1", ["x"], 1.0)],
    )
    assert [r.config for r in compare(a, b)] == ["dense_k5"]


def test_cli_refuses_runs_without_shared_configs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    a = _write(tmp_path, "a", [("dense_k5", "q1", ["x"], 1.0)])
    b = _write(tmp_path, "b", [("sparse_k5", "q1", ["x"], 1.0)])
    assert main([str(a), str(b)]) == 2
    assert "share no config" in capsys.readouterr().err
