"""Unit tests for the custom citation_accuracy metric."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastapi_rag_lab.eval.dataset import GoldenDataset, load_golden_dataset
from fastapi_rag_lab.eval.metrics import citation_accuracy

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"


def test_citation_accuracy_exact_match():
    score = citation_accuracy(["a.md", "b.md"], ["a.md", "b.md"])
    assert score == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_citation_accuracy_partial_match():
    # 1 true positive (a.md), 1 false positive (c.md), 1 false negative (b.md)
    score = citation_accuracy(["a.md", "c.md"], ["a.md", "b.md"])
    assert score["precision"] == pytest.approx(0.5)
    assert score["recall"] == pytest.approx(0.5)
    assert score["f1"] == pytest.approx(0.5)


def test_citation_accuracy_no_overlap():
    score = citation_accuracy(["a.md"], ["b.md"])
    assert score == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_citation_accuracy_empty_returned():
    score = citation_accuracy([], ["a.md", "b.md"])
    assert score == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_citation_accuracy_empty_expected():
    score = citation_accuracy(["a.md"], [])
    assert score == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_citation_accuracy_both_empty():
    score = citation_accuracy([], [])
    assert score == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_citation_accuracy_dedupes_returned():
    """Duplicate IDs in `returned_sources` should not double-count."""
    score = citation_accuracy(["a.md", "a.md"], ["a.md"])
    assert score == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_citation_accuracy_high_precision_low_recall():
    # 1/1 precision, 1/3 recall
    score = citation_accuracy(["a.md"], ["a.md", "b.md", "c.md"])
    assert score["precision"] == pytest.approx(1.0)
    assert score["recall"] == pytest.approx(1 / 3)
    assert score["f1"] == pytest.approx(0.5)


# --- Golden dataset schema tests ---


@pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason="golden_dataset.json not yet committed",
)
def test_golden_dataset_loads_successfully():
    dataset = load_golden_dataset(GOLDEN_PATH)
    assert isinstance(dataset, GoldenDataset)
    assert len(dataset.entries) > 0


@pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason="golden_dataset.json not yet committed",
)
def test_golden_dataset_has_min_entries():
    dataset = load_golden_dataset(GOLDEN_PATH)
    assert len(dataset.entries) >= 30


@pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason="golden_dataset.json not yet committed",
)
def test_golden_dataset_categories_diverse():
    dataset = load_golden_dataset(GOLDEN_PATH)
    categories = {e.category for e in dataset.entries}
    assert len(categories) >= 5, f"only got {categories}"


@pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason="golden_dataset.json not yet committed",
)
def test_golden_dataset_ids_unique():
    dataset = load_golden_dataset(GOLDEN_PATH)
    ids = [e.id for e in dataset.entries]
    assert len(ids) == len(set(ids)), "duplicate ids in golden dataset"


@pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason="golden_dataset.json not yet committed",
)
def test_golden_dataset_expected_sources_resolve():
    """Every expected_source path must exist in the corpus."""
    corpus_root = (
        Path(__file__).parent.parent.parent / "data" / "raw" / "fastapi_docs"
    )
    if not corpus_root.exists():
        pytest.skip("corpus not present (run ingestion first)")

    dataset = load_golden_dataset(GOLDEN_PATH)
    missing: list[str] = []
    for entry in dataset.entries:
        for source in entry.expected_sources:
            if not (corpus_root / source).is_file():
                missing.append(f"{entry.id}: {source}")
    assert not missing, f"missing source files: {missing[:5]}"


def test_load_golden_dataset_accepts_bare_list(tmp_path: Path):
    """Backwards-compat: a bare JSON list should also load."""
    bare = [
        {
            "id": "q_001",
            "query": "test?",
            "reference_answer": "yes",
            "expected_answer_keywords": ["yes"],
            "expected_sources": ["x.md"],
            "category": "factual",
        }
    ]
    path = tmp_path / "bare.json"
    path.write_text(json.dumps(bare))
    ds = load_golden_dataset(path)
    assert len(ds.entries) == 1
    assert ds.entries[0].id == "q_001"
