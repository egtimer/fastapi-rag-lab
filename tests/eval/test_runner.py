"""Unit tests for the eval runner aggregation + threshold logic."""

from __future__ import annotations

from fastapi_rag_lab.eval.runner import (
    THRESHOLDS,
    PerQueryResult,
    _aggregate,
    _check_thresholds,
    _percentile,
)


def _row(**overrides) -> PerQueryResult:
    base = dict(
        id="q",
        query="?",
        category="factual",
        retrieved_source_files=[],
        expected_sources=[],
        generated_answer="",
        reference_answer="",
        citation_precision=1.0,
        citation_recall=1.0,
        citation_f1=1.0,
        faithfulness=0.9,
        answer_relevancy=0.8,
        context_precision=0.85,
        context_recall=0.75,
        latency_ms=100,
        error=None,
    )
    base.update(overrides)
    return PerQueryResult(**base)


def test_aggregate_handles_none_values():
    rows = [
        _row(faithfulness=0.9),
        _row(faithfulness=None, error="generation: timeout"),
        _row(faithfulness=0.7),
    ]
    a = _aggregate(rows)
    # mean only over non-None faithfulness values
    assert a["faithfulness_mean"] == 0.8
    assert a["faithfulness_n"] == 2
    assert a["error_count"] == 1


def test_aggregate_all_none_returns_none_mean():
    rows = [_row(faithfulness=None), _row(faithfulness=None)]
    a = _aggregate(rows)
    assert a["faithfulness_mean"] is None
    assert a["faithfulness_n"] == 0


def test_check_thresholds_passes_when_all_above():
    aggregate = {
        "faithfulness_mean": 0.9,
        "answer_relevancy_mean": 0.8,
        "context_precision_mean": 0.8,
        "context_recall_mean": 0.7,
        "citation_f1_mean": 0.7,
    }
    status = _check_thresholds(aggregate)
    assert all(status.values()), status


def test_check_thresholds_fails_when_below():
    aggregate = {
        "faithfulness_mean": 0.9,
        "answer_relevancy_mean": 0.5,
        "context_precision_mean": 0.8,
        "context_recall_mean": 0.4,
        "citation_f1_mean": 0.7,
    }
    status = _check_thresholds(aggregate)
    assert status["faithfulness"] is True
    assert status["answer_relevancy"] is False
    assert status["context_recall"] is False


def test_check_thresholds_marks_none_as_failure():
    """If a metric had zero successful samples, treat that as a failure."""
    aggregate = {
        "faithfulness_mean": None,
        "answer_relevancy_mean": 0.9,
        "context_precision_mean": 0.9,
        "context_recall_mean": 0.9,
        "citation_f1_mean": 0.9,
    }
    status = _check_thresholds(aggregate)
    assert status["faithfulness"] is False


def test_percentile_basic():
    values = [10, 20, 30, 40, 50]
    assert _percentile(values, 50) == 30.0
    assert _percentile(values, 100) == 50.0
    assert _percentile(values, 0) == 10.0


def test_percentile_empty():
    assert _percentile([], 50) is None


def test_thresholds_match_documented_values():
    """If thresholds drift, ADR 005 also needs updating -- pin them here."""
    assert THRESHOLDS == {
        "faithfulness": 0.75,
        "answer_relevancy": 0.70,
        "context_precision": 0.70,
        "context_recall": 0.65,
        "citation_f1": 0.60,
    }
