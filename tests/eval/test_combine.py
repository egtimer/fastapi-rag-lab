"""Tests for pooling calibration batches (fastapi_rag_lab.eval.judge.combine)."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

from fastapi_rag_lab.eval.judge import combine

if TYPE_CHECKING:
    from pathlib import Path


def _write_batch(
    root: Path,
    items: str,
    run: str,
    first: list[str],
    second: list[str],
    judge: list[str],
) -> None:
    (root / items).mkdir(parents=True)
    (root / run).mkdir(parents=True)
    for name, labels in (("labels.csv", first), ("labels_second.csv", second)):
        with (root / items / name).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["item_id", "label", "note"])
            w.writeheader()
            for i, label in enumerate(labels):
                w.writerow({"item_id": f"j-{i:02d}", "label": label, "note": ""})
    with (root / run / "judgements.jsonl").open("w", encoding="utf-8") as fh:
        for i, label in enumerate(judge):
            fh.write(json.dumps({"item_id": f"j-{i:02d}", "label": label}) + "\n")


S, U = "supported", "unsupported"


def test_regenerates_the_committed_n100_report(tmp_path: Path) -> None:
    """The defaults rebuild reports/2026-11-16-judge-agreement-n100 byte for byte."""
    committed = combine.ROOT / combine.N100_OUT
    out = tmp_path / "agreement.json"
    assert combine.main(["--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8")


def test_prefixes_keep_batches_with_the_same_ids_apart(tmp_path: Path) -> None:
    _write_batch(
        tmp_path, "a/items", "a/judge", [S, U, S, U], [S, U, S, U], [S, U, U, U]
    )
    _write_batch(
        tmp_path, "b/items", "b/judge", [S, U, S, U], [S, U, S, U], [S, U, S, U]
    )
    report = combine.combined_report(
        tmp_path, {"a": ("a/items", "a/judge"), "b": ("b/items", "b/judge")}
    )
    c = report["comparisons"]
    assert c["human_vs_second"]["n"] == 8
    assert c["human_vs_sonnet_v1"]["n"] == 8
    assert c["human_vs_sonnet_v1"]["accuracy"]["value"] == pytest.approx(7 / 8)
    assert report["kappa_ci_width"]["human_vs_second"]["dev_only"] is None


def test_paired_difference_is_zero_when_judge_matches_second() -> None:
    human = {f"x:{i}": lab for i, lab in enumerate([S, U, S, U, U, S])}
    second = {f"x:{i}": lab for i, lab in enumerate([S, U, U, U, U, S])}
    d = combine.paired_difference(human, second, dict(second), resamples=200)
    assert d["n"] == 6
    assert d["value"] == pytest.approx(0.0)
    assert d["ci_low"] == pytest.approx(0.0)
    assert d["ci_high"] == pytest.approx(0.0)
    assert d["share_le_zero"] == 1.0


def test_paired_difference_skips_items_not_labelled_by_all_three() -> None:
    human = {"x:0": S, "x:1": U, "x:2": S, "x:3": U, "x:4": "unsure"}
    second = {"x:0": S, "x:1": U, "x:2": S, "x:3": U, "x:4": S}
    judge = {"x:0": S, "x:1": U, "x:2": U, "x:3": U}
    d = combine.paired_difference(human, second, judge, resamples=200)
    assert d["n"] == 4
    assert d["value"] == pytest.approx(1.0 - 0.5)


def test_batch_argument_needs_items_and_judge_run() -> None:
    assert combine._batch_arg("dev=r/items,r/judge") == ("dev", ("r/items", "r/judge"))
    with pytest.raises(Exception, match="NAME=ITEMS_RUN,JUDGE_RUN"):
        combine._batch_arg("dev=r/items")


def test_cli_with_explicit_batches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_batch(tmp_path, "i", "j", [S, U, S, U], [S, U, S, S], [S, U, S, U])
    out = tmp_path / "out" / "agreement.json"
    rc = combine.main(["--root", str(tmp_path), "--batch", "k=i,j", "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["batches"] == {"k": ["i", "j"]}
    assert report["comparisons"]["human_vs_sonnet_v1"]["kappa"][
        "value"
    ] == pytest.approx(1.0)
    assert "paired difference" in capsys.readouterr().out
