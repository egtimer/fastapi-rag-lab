"""Offline tests for the label precision sample."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from fastapi_rag_lab.eval.label_audit import main

if TYPE_CHECKING:
    from pathlib import Path


def _questions(tmp_path: Path) -> Path:
    path = tmp_path / "questions.jsonl"
    rows = [
        {
            "id": f"so_{i}",
            "split": "dev" if i % 2 else "test",
            "query": f"question {i}",
            "link": f"https://stackoverflow.com/questions/{i}",
            "expected_sources": ["tutorial/a.md", "tutorial/b.md"][: 1 + i % 2],
            "label_answer_ids": [100 + i],
        }
        for i in range(10)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _read(out: Path) -> list[dict]:
    with (out / "labels.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_every_pair_once_with_the_first_n_marked(tmp_path: Path):
    out = tmp_path / "audit"
    assert (
        main(["--questions", str(_questions(tmp_path)), "--out", str(out), "--n", "4"])
        == 0
    )
    rows = _read(out)
    assert len(rows) == 15  # 5 questions with one label, 5 with two
    assert len({r["pair_id"] for r in rows}) == 15
    assert [r["in_sample"] for r in rows].count("yes") == 4
    assert all(r["in_sample"] == "yes" for r in rows[:4])
    assert all(r["verdict"] == "" for r in rows)


def test_same_seed_same_order(tmp_path: Path):
    q = _questions(tmp_path)
    main(
        ["--questions", str(q), "--out", str(tmp_path / "a"), "--seed", "3", "--n", "5"]
    )
    main(
        ["--questions", str(q), "--out", str(tmp_path / "b"), "--seed", "3", "--n", "5"]
    )
    assert _read(tmp_path / "a") == _read(tmp_path / "b")


def test_refuses_a_sample_larger_than_the_pairs(tmp_path: Path):
    q = _questions(tmp_path)
    assert main(["--questions", str(q), "--out", str(tmp_path / "c"), "--n", "16"]) == 2
