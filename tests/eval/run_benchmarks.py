#!/usr/bin/env python
"""Run retrieval benchmarks (retrieval-only metrics, no LLM judge).

Usage (needs Qdrant + Ollama running):
    OLLAMA_HOST=http://localhost:11434 uv run python tests/eval/run_benchmarks.py \
        [--dataset PATH] [--strategies sparse,dense,hybrid,hybrid_rerank] \
        [--top-k 3,5,10] [--run-id NAME] [--out-root reports] \
        [--plots-dir DIR] [--warmup 1]

Writes <out-root>/<run-id>/benchmark.json (with provenance) and analysis.json
(bootstrap CIs, 10000 resamples, seed 0). Refuses an existing run directory.
"""

from __future__ import annotations

from fastapi_rag_lab.eval.benchmark_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
