"""Offline tests for the regression gate, using tiny synthetic reports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi_rag_lab.eval.check_regression import main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

N = 40


def _write(
    root: Path,
    name: str,
    recall: list[float],
    *,
    sha: str = "abc",
    config: str = "hybrid_k5",
) -> Path:
    d = root / name
    d.mkdir()
    rows = [
        {
            "config_label": config,
            "query_id": f"q{i}",
            "recall_at_k": r,
            "mrr": r,
            "latency_ms": 10.0,
        }
        for i, r in enumerate(recall)
    ]
    report = {"per_query": rows, "provenance": {"dataset": {"sha256": sha}}}
    (d / "benchmark.json").write_text(json.dumps(report))
    return d


def _run(ref: Path, cur: Path, *extra: str) -> int:
    return main(["--reference", str(ref), "--current", str(cur), *extra])


def test_identical_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = [1.0, 0.0] * (N // 2)
    ref, cur = _write(tmp_path, "r", base), _write(tmp_path, "c", base)
    assert _run(ref, cur) == 0
    assert "Result: pass" in capsys.readouterr().out


def test_big_drop_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ref = _write(tmp_path, "r", [1.0] * N)
    cur = _write(tmp_path, "c", [1.0] * (N // 2) + [0.0] * (N // 2))
    assert _run(ref, cur) == 1
    assert "Result: fail" in capsys.readouterr().out


def test_small_drop_with_ci_touching_zero_passes(tmp_path: Path) -> None:
    ref = _write(tmp_path, "r", [1.0] * N)
    cur = _write(tmp_path, "c", [0.0] + [1.0] * (N - 1))  # mean drop 0.025
    assert _run(ref, cur) == 0


def test_dataset_sha_mismatch_exits_2(tmp_path: Path) -> None:
    ref = _write(tmp_path, "r", [1.0] * N, sha="aaa")
    cur = _write(tmp_path, "c", [1.0] * N, sha="bbb")
    assert _run(ref, cur) == 2


def test_missing_config_exits_2(tmp_path: Path) -> None:
    ref = _write(tmp_path, "r", [1.0] * N)
    cur = _write(tmp_path, "c", [1.0] * N, config="dense_k5")
    assert _run(ref, cur) == 2


def test_query_id_mismatch_exits_2(tmp_path: Path) -> None:
    ref = _write(tmp_path, "r", [1.0] * N)
    cur = _write(tmp_path, "c", [1.0] * (N - 1))
    assert _run(ref, cur) == 2


def test_step_summary_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    ref = _write(tmp_path, "r", [1.0] * N)
    cur = _write(tmp_path, "c", [1.0] * N)
    assert _run(ref, cur) == 0
    assert "Result: pass" in summary.read_text()
