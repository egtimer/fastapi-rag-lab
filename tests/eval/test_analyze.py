import json
from pathlib import Path

import pytest

from fastapi_rag_lab.eval.analyze import analyze_report, main

# per-query values: (recall_at_k, mrr, ndcg_at_k, citation_f1)
VALUES = {
    "dense": [(1.0, 1.0, 1.0, 0.5), (0.0, 0.0, 0.0, 0.0), (1.0, 0.5, 0.6, 0.4)],
    "hybrid": [(1.0, 1.0, 1.0, 0.5), (1.0, 0.5, 0.6, 0.2), (1.0, 1.0, 1.0, 0.5)],
}
METRICS = ("recall_at_k", "mrr", "ndcg_at_k", "citation_f1")


def _report(tamper: bool = False) -> dict:
    aggregates, per_query = [], []
    for strategy, rows in VALUES.items():
        label = f"{strategy}_k5"
        agg = {"config_label": label, "strategy": strategy, "top_k": 5}
        for j, m in enumerate(METRICS):
            agg[f"{m}_mean"] = sum(r[j] for r in rows) / len(rows)
        aggregates.append(agg)
        for i, r in enumerate(rows):
            per_query.append(
                {
                    "config_label": label,
                    "query_id": f"q{i}",
                    **dict(zip(METRICS, r, strict=True)),
                }
            )
    if tamper:
        aggregates[0]["mrr_mean"] += 0.1
    return {"configs": [], "aggregates": aggregates, "per_query": per_query}


def test_analyze_means_and_comparisons() -> None:
    res = analyze_report(_report(), n_resamples=200)
    dense = next(c for c in res["configs"] if c["config_label"] == "dense_k5")
    assert dense["n"] == 3
    assert dense["recall_at_k"]["mean"] == pytest.approx(2 / 3)
    assert dense["mrr"]["mean"] == pytest.approx(0.5)
    (cmp_,) = [c for c in res["comparisons"] if c["metric"] == "recall_at_k"]
    assert (cmp_["a"], cmp_["b"], cmp_["top_k"]) == ("dense", "hybrid", 5)
    assert cmp_["mean_diff"] == pytest.approx(1 / 3)
    assert (cmp_["wins_b"], cmp_["wins_a"], cmp_["ties"]) == (1, 0, 2)


def test_analyze_rejects_mismatched_aggregates() -> None:
    with pytest.raises(ValueError, match="stored aggregate"):
        analyze_report(_report(tamper=True), n_resamples=50)


def test_cli_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "b.json"
    src.write_text(json.dumps(_report()))
    out = tmp_path / "a.json"
    assert main([str(src), "--out", str(out), "--resamples", "100"]) == 0
    data = json.loads(out.read_text())
    assert data["resamples"] == 100
    assert len(data["input_sha256"]) == 64
    assert "dense_k5" in capsys.readouterr().out
