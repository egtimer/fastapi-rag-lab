"""Offline tests: benchmark retriever parity with the API, provenance, CLI."""

from __future__ import annotations

import socket

import pytest

from fastapi_rag_lab.eval import benchmark_cli, benchmarks, provenance
from fastapi_rag_lab.eval.benchmarks import RetrieverConfig, build_retriever
from fastapi_rag_lab.eval.dataset import GoldenEntry
from fastapi_rag_lab.retrieval import hybrid as hybrid_mod
from fastapi_rag_lab.retrieval.hybrid import HybridRetriever
from fastapi_rag_lab.retrieval.reranker import RERANKER_REVISION
from fastapi_rag_lab.retrieval.types import RerankedResult, RetrievalCandidate


def _candidate(i: int, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=f"c{i}",
        parent_id=f"p{i}",
        score=score,
        text=f"child {i}",
        metadata={"parent_text": f"parent {i}", "source_file": f"doc{i}.md"},
    )


class FakeRetriever:
    def __init__(self, order: list[int]):
        self.order = order
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 30) -> list[RetrievalCandidate]:
        self.calls.append((query, top_k))
        return [
            _candidate(i, 1.0 / (rank + 1)) for rank, i in enumerate(self.order[:top_k])
        ]


class FakeReranker:
    def rerank(self, query, candidates, top_k=5) -> list[RerankedResult]:
        return [
            RerankedResult(
                chunk_id=c.chunk_id,
                parent_id=c.parent_id,
                parent_text=c.metadata["parent_text"],
                chunk_text=c.text,
                rerank_score=float(n),
                rrf_score=c.rrf_score,
                metadata=c.metadata,
            )
            for n, c in enumerate(reversed(candidates[:top_k]))
        ]


def _parts():
    return FakeRetriever([1, 2, 3, 4, 5, 6]), FakeRetriever([1, 3, 2, 6, 5, 4])


@pytest.mark.parametrize("strategy", ["hybrid", "hybrid_rerank"])
def test_configured_retriever_matches_hybrid_retriever(strategy):
    dense, sparse = _parts()
    reranker = FakeReranker()
    cfg = RetrieverConfig(strategy=strategy, top_k=3, rerank_top_n=5)
    configured = build_retriever(cfg, dense, sparse, reranker)
    assert isinstance(configured.hybrid, HybridRetriever)

    got = configured.retrieve("q")
    direct = HybridRetriever(
        strategy=strategy, dense=dense, sparse=sparse, reranker=reranker
    ).retrieve("q", dense_k=30, sparse_k=30, fusion_k=5, final_k=3)
    assert [r.chunk_id for r in got] == [r.chunk_id for r in direct]
    assert len(got) == 3
    assert dense.calls[0] == ("q", 30)
    assert sparse.calls[0] == ("q", 30)


def test_rrf_k_is_passed_through(monkeypatch):
    seen = []
    real = hybrid_mod.reciprocal_rank_fusion

    def spy(**kwargs):
        seen.append(kwargs["k"])
        return real(**kwargs)

    monkeypatch.setattr(hybrid_mod, "reciprocal_rank_fusion", spy)
    dense, sparse = _parts()
    r = HybridRetriever(strategy="hybrid", dense=dense, sparse=sparse)
    r.retrieve("q")
    r.retrieve("q", rrf_k=7)
    cfg = RetrieverConfig(strategy="hybrid", top_k=3, rrf_k=11)
    build_retriever(cfg, dense, sparse).retrieve("q")
    assert seen == [60, 7, 11]


def test_sparse_strategy_uses_bm25_only():
    dense, sparse = _parts()
    cfg = RetrieverConfig(strategy="sparse", top_k=3)
    out = build_retriever(cfg, dense, sparse).retrieve("q")
    assert [r.chunk_id for r in out] == ["c1", "c3", "c2"]
    assert sparse.calls == [("q", 3)]
    assert dense.calls == []
    assert cfg.label == "sparse_k3"


def test_sparse_requires_sparse_retriever():
    with pytest.raises(ValueError):
        build_retriever(RetrieverConfig(strategy="sparse", top_k=3), FakeRetriever([]))


def test_run_dir_refused_if_exists(tmp_path):
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        benchmark_cli.prepare_run_dir(tmp_path, "taken")
    assert benchmark_cli.prepare_run_dir(tmp_path, "new") == tmp_path / "new"
    assert benchmark_cli.main(["--out-root", str(tmp_path), "--run-id", "taken"]) == 2


def _closed_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_service_probes_degrade_to_null_with_reason():
    url = _closed_url()
    q = provenance.qdrant_info(url)
    assert q["server_version"] is None and q["point_count"] is None
    assert q["reason"]
    o = provenance.ollama_info(url)
    assert o["embedding_model_digest"] is None and o["reason"]


def test_missing_files_degrade(tmp_path):
    c = provenance.corpus_info(tmp_path / "nope.json")
    assert c["manifest_sha256"] is None and c["reason"]
    assert c["pinned_commit"]
    d = provenance.dataset_info(tmp_path / "nope.json", 3)
    assert d["sha256"] is None and d["reason"] and d["entry_count"] == 3
    g = provenance.git_info(tmp_path / "does-not-exist")
    assert g["commit"] is None and g["reason"]


def test_collect_provenance_shape(tmp_path, monkeypatch):
    url = _closed_url()
    monkeypatch.setenv("QDRANT_URL", url)
    monkeypatch.setenv("OLLAMA_HOST", url)
    ds = tmp_path / "d.json"
    ds.write_text("{}")
    p = provenance.collect_provenance(
        dataset_path=ds, entry_count=1, argv=["x", "--y"], started_at="a"
    )
    assert p["command"] == "x --y"
    assert p["dataset"]["sha256"]
    assert p["reranker_model"]
    assert p["reranker_revision"] == RERANKER_REVISION
    assert {"git", "corpus", "qdrant", "ollama", "packages", "hardware"} <= set(p)
    assert p["packages"]["python"]
    assert p["hardware"]["logical_cores"]


def test_cli_defaults():
    args = benchmark_cli.build_parser().parse_args([])
    assert args.strategies == ["dense", "hybrid", "hybrid_rerank"]
    assert args.top_k == [3, 5, 10]
    assert args.warmup == 1
    assert args.plots_dir is None
    assert args.run_id is None
    assert str(args.dataset) == "tests/eval/golden_dataset.json"
    assert str(args.out_root) == "reports"


def test_cli_parses_options():
    args = benchmark_cli.build_parser().parse_args(
        [
            "--dataset", "x/y.json",
            "--strategies", "sparse,dense",
            "--top-k", "1,2",
            "--run-id", "r",
            "--plots-dir", "p",
            "--warmup", "0",
        ]
    )  # fmt: skip
    assert args.strategies == ["sparse", "dense"]
    assert args.top_k == [1, 2]
    assert args.run_id == "r" and args.warmup == 0
    assert benchmark_cli.default_run_id(args.dataset).endswith("-y-benchmark")


@pytest.mark.parametrize("bad", [["--strategies", "bogus"], ["--top-k", "a"]])
def test_cli_rejects_bad_values(bad):
    with pytest.raises(SystemExit):
        benchmark_cli.build_parser().parse_args(bad)


# --- checkpoint and resume -------------------------------------------------


def _row(label: str, qid: str) -> benchmarks.PerQueryBenchmark:
    return benchmarks.PerQueryBenchmark(
        config_label=label,
        query_id=qid,
        query=f"query {qid}",
        category="factual",
        retrieved_sources=["a.md"],
        expected_sources=["a.md"],
        citation_f1=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        recall_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        latency_ms=12.5,
    )


def _entries(*ids: str) -> list[GoldenEntry]:
    return [GoldenEntry(id=i, query=f"query {i}", category="factual") for i in ids]


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    cfg = RetrieverConfig(strategy="dense", top_k=5)
    rows = [_row("dense_k5", "q1"), _row("dense_k5", "q2")]

    benchmarks._append_checkpoint(path, "sha", cfg, rows)
    done = benchmarks._load_checkpoint(path, "sha", _entries("q1", "q2"))

    assert list(done) == ["dense_k5"]
    assert done["dense_k5"] == rows


def test_missing_checkpoint_means_nothing_done(tmp_path):
    assert benchmarks._load_checkpoint(None, "sha", _entries("q1")) == {}
    assert benchmarks._load_checkpoint(tmp_path / "none", "sha", _entries("q1")) == {}


def test_checkpoint_from_another_dataset_is_refused(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    cfg = RetrieverConfig(strategy="dense", top_k=5)
    benchmarks._append_checkpoint(path, "sha-a", cfg, [_row("dense_k5", "q1")])

    with pytest.raises(ValueError, match="different dataset"):
        benchmarks._load_checkpoint(path, "sha-b", _entries("q1"))
    with pytest.raises(ValueError, match="do not match"):
        benchmarks._load_checkpoint(path, "sha-a", _entries("q1", "q2"))


def test_resume_accepts_only_an_unfinished_run_dir(tmp_path):
    (tmp_path / "unfinished").mkdir()
    finished = tmp_path / "finished"
    finished.mkdir()
    (finished / "benchmark.json").write_text("{}")

    assert (
        benchmark_cli.prepare_run_dir(tmp_path, "unfinished", resume=True)
        == tmp_path / "unfinished"
    )
    with pytest.raises(FileExistsError):
        benchmark_cli.prepare_run_dir(tmp_path, "finished", resume=True)
    with pytest.raises(FileExistsError):
        benchmark_cli.prepare_run_dir(tmp_path, "unfinished")


def _mismatch(match, **extra):
    return {
        "index": {"ollama_version": "0.20.7"},
        "query": {"ollama_version": "0.35.1"},
        "match": match,
        **extra,
    }


def test_cli_refuses_runtime_mismatch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark_cli, "check_runtime_matches_index", lambda *a: _mismatch(False)
    )

    def boom(*a, **k):
        raise AssertionError("benchmark must not run")

    monkeypatch.setattr(benchmark_cli, "run_benchmark", boom)
    assert benchmark_cli.main(["--out-root", str(tmp_path), "--run-id", "r"]) == 2
    err = capsys.readouterr().err
    assert "0.20.7" in err and "0.35.1" in err and "--allow-runtime-mismatch" in err


def test_cli_allows_runtime_mismatch_and_records_it(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark_cli, "check_runtime_matches_index", lambda *a: _mismatch(False)
    )
    seen = {}

    class StopError(Exception):
        pass

    def fake_run(*a, **k):
        seen.update(k["runtime_check"])
        raise StopError

    monkeypatch.setattr(benchmark_cli, "run_benchmark", fake_run)
    with pytest.raises(StopError):
        benchmark_cli.main(
            ["--out-root", str(tmp_path), "--run-id", "r", "--allow-runtime-mismatch"]
        )
    assert seen["match"] is False and seen["allowed_mismatch"] is True


def test_cli_continues_when_runtime_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark_cli,
        "check_runtime_matches_index",
        lambda *a: _mismatch(None, reason="index has no runtime fields"),
    )

    class StopError(Exception):
        pass

    def fake_run(*a, **k):
        raise StopError

    monkeypatch.setattr(benchmark_cli, "run_benchmark", fake_run)
    with pytest.raises(StopError):
        benchmark_cli.main(["--out-root", str(tmp_path), "--run-id", "r"])


def test_ollama_info_has_version_field():
    o = provenance.ollama_info(_closed_url())
    assert o["ollama_version"] is None and "version:" in o["reason"]
