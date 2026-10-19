"""Command-line driver for retrieval benchmarks (see tests/eval/run_benchmarks.py)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from fastapi_rag_lab.eval.analyze import analyze_report, format_markdown
from fastapi_rag_lab.eval.benchmarks import (
    RetrieverConfig,
    print_benchmark_summary,
    run_benchmark,
)
from fastapi_rag_lab.ingest.embedder import default_ollama_host
from fastapi_rag_lab.ingest.runtime_check import (
    check_runtime_matches_index,
    format_runtime,
)
from fastapi_rag_lab.ingest.store import COLLECTION_NAME

ALL_STRATEGIES = ("sparse", "dense", "hybrid", "hybrid_rerank")
DEFAULT_STRATEGIES = "dense,hybrid,hybrid_rerank"
DEFAULT_DATASET = Path("tests/eval/golden_dataset.json")
RESAMPLES = 10_000
SEED = 0


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _strategies(value: str) -> list[str]:
    items = _csv(value)
    bad = [s for s in items if s not in ALL_STRATEGIES]
    if bad or not items:
        raise argparse.ArgumentTypeError(
            f"unknown strategies {bad}; choose from {','.join(ALL_STRATEGIES)}"
        )
    return items


def _top_ks(value: str) -> list[int]:
    try:
        ks = [int(v) for v in _csv(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --top-k {value!r}") from exc
    if not ks or any(k < 1 for k in ks):
        raise argparse.ArgumentTypeError("--top-k needs positive integers")
    return ks


def default_run_id(dataset: Path) -> str:
    return f"{datetime.now(UTC):%Y-%m-%d}-{dataset.stem}-benchmark"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run retrieval benchmarks.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--strategies", type=_strategies, default=_strategies(DEFAULT_STRATEGIES)
    )
    p.add_argument("--top-k", type=_top_ks, default=[3, 5, 10])
    p.add_argument(
        "--run-id", default=None, help="default: <UTC date>-<dataset stem>-benchmark"
    )
    p.add_argument("--out-root", type=Path, default=Path("reports"))
    p.add_argument("--plots-dir", type=Path, default=None, help="no plots unless given")
    p.add_argument(
        "--warmup", type=int, default=1, help="unmeasured queries per config"
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue an unfinished run: configs in its checkpoint are not rerun",
    )
    p.add_argument(
        "--allow-runtime-mismatch",
        action="store_true",
        help="run even if the Ollama runtime differs from the one that built the index",
    )
    return p


CHECKPOINT = "checkpoint.jsonl"


def prepare_run_dir(out_root: Path, run_id: str, *, resume: bool = False) -> Path:
    """Return the run directory path, refusing one that already exists.

    With ``resume``, an existing directory is accepted while it has no
    ``benchmark.json`` (the run did not finish); a finished run is still refused.
    """
    run_dir = out_root / run_id
    if run_dir.exists() and not (resume and not (run_dir / "benchmark.json").exists()):
        raise FileExistsError(f"run directory already exists: {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    run_id = args.run_id or default_run_id(args.dataset)
    try:
        run_dir = prepare_run_dir(args.out_root, run_id, resume=args.resume)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        runtime_check = check_runtime_matches_index(
            qdrant_url, COLLECTION_NAME, default_ollama_host()
        )
    except Exception as exc:
        print(f"error: cannot check the embedding runtime: {exc}", file=sys.stderr)
        return 2
    if runtime_check["match"] is False:
        msg = (
            f"embedding runtime mismatch: index built with "
            f"{format_runtime(runtime_check['index'])}; "
            f"querying with {format_runtime(runtime_check['query'])}"
        )
        if not args.allow_runtime_mismatch:
            print(
                f"error: {msg} (pass --allow-runtime-mismatch to run anyway)",
                file=sys.stderr,
            )
            return 2
        logging.warning("%s; continuing (--allow-runtime-mismatch)", msg)
        runtime_check["allowed_mismatch"] = True
    elif runtime_check["match"] is None:
        logging.warning(
            "embedding runtime not verified: %s", runtime_check.get("reason")
        )

    configs = [
        RetrieverConfig(strategy=s, top_k=k)
        for s in args.strategies
        for k in args.top_k
    ]
    command_argv = ["run_benchmarks.py", *(sys.argv[1:] if argv is None else argv)]
    benchmark_path = run_dir / "benchmark.json"
    results = run_benchmark(
        configs,
        args.dataset,
        benchmark_path,
        warmup=args.warmup,
        argv=command_argv,
        checkpoint_path=run_dir / CHECKPOINT,
        runtime_check=runtime_check,
    )
    print_benchmark_summary(results)

    report = json.loads(benchmark_path.read_text(encoding="utf-8"))
    analysis = analyze_report(report, n_resamples=RESAMPLES, seed=SEED)
    out = {
        "input_path": str(benchmark_path),
        "input_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "resamples": RESAMPLES,
        "seed": SEED,
        "alpha": 0.05,
        "command": " ".join(command_argv),
        **analysis,
    }
    (run_dir / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")
    print(format_markdown(analysis))
    # benchmark.json holds every row now, including the resumed ones
    (run_dir / CHECKPOINT).unlink(missing_ok=True)

    if args.plots_dir is not None:
        from fastapi_rag_lab.eval.plotting import save_all_plots

        for path in save_all_plots(results, args.plots_dir):
            logging.info("Saved plot: %s", path)
    return 0
