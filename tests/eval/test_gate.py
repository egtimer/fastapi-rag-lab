"""The judge-change gate: false passes on human-unsupported answers must not rise."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastapi_rag_lab.eval.judge import gate

REPORTS = Path(__file__).resolve().parents[2] / "reports"
ITEMS = REPORTS / "2026-10-30-judge-items-so-dev-50"
HAIKU_GATE = REPORTS / "2026-11-19-judge-gate-haiku-structured"
PLAIN = [
    "2026-10-30-judge-haiku-so-dev-50",
    "2026-11-18-judge-haiku-plain-r1-so-dev-50",
    "2026-11-18-judge-haiku-plain-r2-so-dev-50",
    "2026-11-18-judge-haiku-plain-r3-so-dev-50",
]
STRUCTURED = [
    "2026-11-10-judge-haiku-structured-so-dev-50",
    "2026-11-18-judge-haiku-structured-r1-so-dev-50",
    "2026-11-18-judge-haiku-structured-r2-so-dev-50",
    "2026-11-18-judge-haiku-structured-r3-so-dev-50",
]
SONNET = [
    "2026-10-30-judge-sonnet-so-dev-50",
    "2026-11-10-judge-sonnet-retest-so-dev-50",
    "2026-11-16-judge-sonnet-retest2-so-dev-50",
    "2026-11-16-judge-sonnet-retest3-so-dev-50",
    "2026-11-16-judge-sonnet-retest4-so-dev-50",
    "2026-11-16-judge-sonnet-retest5-so-dev-50",
]

HUMAN = {"a": "unsupported", "b": "unsupported", "c": "unsupported", "d": "supported"}


def test_items_need_a_verdict_from_every_run() -> None:
    runs = {
        "x": {"a": "supported", "b": "unparsed", "c": "unsupported", "d": "supported"},
        "y": {"a": "unsupported", "b": "unsupported", "c": "error"},
    }
    assert gate.compared_items(HUMAN, runs) == ["a"]


def test_gate_passes_at_the_baseline_ceiling_and_fails_above() -> None:
    baseline = {
        "b1": {"a": "supported", "b": "unsupported", "c": "unsupported"},
        "b2": {"a": "unsupported", "b": "unsupported", "c": "unsupported"},
    }
    same = gate.gate_report(HUMAN, baseline, {"c1": baseline["b1"]})
    assert same["n_items"] == 3
    assert same["baseline_range"] == [0.0, pytest.approx(1 / 3)]
    assert same["passed"]
    worse = {
        "c1": {"a": "supported", "b": "supported", "c": "unsupported"},
        "c2": {"a": "supported", "b": "unsupported", "c": "unsupported"},
    }
    report = gate.gate_report(HUMAN, baseline, worse)
    assert report["candidate_mean"] == pytest.approx(0.5)
    assert not report["passed"]


def test_gate_rejects_bad_input() -> None:
    run = {"a": "unsupported"}
    with pytest.raises(ValueError, match="at least one"):
        gate.gate_report(HUMAN, {"b": run}, {})
    with pytest.raises(ValueError, match="both"):
        gate.gate_report(HUMAN, {"r": run}, {"r": run})
    with pytest.raises(ValueError, match="no item"):
        gate.gate_report(HUMAN, {"b": {"a": "unparsed"}}, {"c": run})


def _args(baseline: list[str], candidate: list[str], *extra: str) -> list[str]:
    args = ["--items", str(ITEMS)]
    for name in baseline:
        args += ["--baseline", str(REPORTS / name)]
    for name in candidate:
        args += ["--candidate", str(REPORTS / name)]
    return [*args, *extra]


def test_forced_schema_on_haiku_fails_the_gate(tmp_path: Path) -> None:
    """The committed gate report is what the code produces from the committed runs."""
    out = tmp_path / "gate.json"
    assert gate.main(_args(PLAIN, STRUCTURED, "--out", str(out))) == 1
    report = json.loads(out.read_text())
    assert report == json.loads((HAIKU_GATE / "gate.json").read_text())
    assert report["n_items"] == 28
    assert [r["false_pass"] for r in report["baseline"].values()] == [4, 6, 3, 4]
    assert [r["false_pass"] for r in report["candidate"].values()] == [8, 9, 7, 6]


def test_reruns_of_the_reported_judge_pass_the_gate() -> None:
    """Three Sonnet re-runs against the other three: no change, so no failure."""
    assert gate.main(_args(SONNET[:3], SONNET[3:])) == 0


def test_cli_refuses_mixed_or_identical_configs() -> None:
    mixed = _args([PLAIN[1], STRUCTURED[1]], [STRUCTURED[2]])
    assert gate.main(mixed) == 2
    identical = _args(PLAIN[1:2], PLAIN[2:3])
    assert gate.main(identical) == 2
