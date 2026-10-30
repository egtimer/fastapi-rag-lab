"""Generate the answers the judge is calibrated on, as the API would give them.

A seeded random sample of questions from a dataset (the Stack Overflow dev
split by default) goes through the API's own path: ``HybridRetriever`` with the
default strategy and top_k=5, ``build_prompt`` with the parent texts, and the
API's ``OllamaClient`` (``gemma3:4b``). Each finished item is appended to
``items.jsonl`` at once, so ``--resume`` continues an interrupted run. Free
(local models only); about 85 s per answer on a 4-core CPU.

    QDRANT_URL=http://localhost:6333 OLLAMA_HOST=http://localhost:11434 \\
      uv run python -m fastapi_rag_lab.eval.judge.answers \\
        --run-id 2026-10-30-judge-items-so-dev-50 --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from fastapi_rag_lab.api.llm.ollama_client import OllamaClient
from fastapi_rag_lab.api.prompt import build_prompt
from fastapi_rag_lab.eval.dataset import load_golden_dataset
from fastapi_rag_lab.eval.provenance import collect_provenance
from fastapi_rag_lab.ingest.embedder import default_ollama_host
from fastapi_rag_lab.ingest.runtime_check import check_runtime_matches_index
from fastapi_rag_lab.ingest.store import COLLECTION_NAME

DEFAULT_DATASET = Path("tests/eval/stackoverflow/dev.json")
TOP_K = 5  # the API's default top_k
# The API's 120 s timeout is too short for a cold model load on CPU; it does
# not change the answer.
GENERATION_TIMEOUT_S = 900.0


def sample_entries(entries: list, n: int, seed: int) -> list:
    """``n`` entries drawn without replacement, independent of the file order."""
    ordered = sorted(entries, key=lambda e: e.id)
    if not 0 < n <= len(ordered):
        raise ValueError(f"n must be between 1 and {len(ordered)}, got {n}")
    return random.Random(seed).sample(ordered, n)


def read_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def model_digest(host: str, model: str) -> str | None:
    resp = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=5)
    resp.raise_for_status()
    for m in resp.json().get("models", []):
        if m.get("name") == model or m.get("model") == model:
            return m.get("digest")
    return None


def make_item(item_id: str, entry, retriever, client: OllamaClient) -> dict[str, Any]:
    t0 = time.perf_counter()
    results = retriever.retrieve(entry.query, final_k=TOP_K)
    t1 = time.perf_counter()
    prompt = build_prompt(
        entry.query, [(i + 1, r.parent_text) for i, r in enumerate(results)]
    )
    answer = asyncio.run(client.generate(prompt))
    t2 = time.perf_counter()
    return {
        "item_id": item_id,
        "question_id": entry.id,
        "question": entry.query,
        "expected_sources": entry.expected_sources,
        "sources": [
            {
                "source_file": r.metadata.get("source_file", ""),
                "heading_path": r.metadata.get("heading_path", []),
                "parent_text": r.parent_text,
            }
            for r in results
        ],
        "answer": answer,
        "retrieval_s": round(t1 - t0, 3),
        "generation_s": round(t2 - t1, 1),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-root", type=Path, default=Path("reports"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    run_dir = args.out_root / args.run_id
    items_path = run_dir / "items.jsonl"
    if run_dir.exists() and not args.resume:
        print(
            f"error: {run_dir} exists (pass --resume to continue it)", file=sys.stderr
        )
        return 2

    host = default_ollama_host()
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    runtime = check_runtime_matches_index(qdrant_url, COLLECTION_NAME, host)
    if runtime["match"] is not True:
        print(f"error: embedding runtime not verified: {runtime}", file=sys.stderr)
        return 2

    from fastapi_rag_lab.retrieval.hybrid import HybridRetriever

    entries = load_golden_dataset(args.dataset).entries
    sample = sample_entries(entries, args.n, args.seed)
    client = OllamaClient(timeout=GENERATION_TIMEOUT_S)
    retriever = HybridRetriever()
    run_dir.mkdir(parents=True, exist_ok=True)

    run_json = run_dir / "run.json"
    if not run_json.exists():
        command = ["answers.py", *(sys.argv[1:] if argv is None else argv)]
        meta = {
            "run_id": args.run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "seed": args.seed,
            "n": args.n,
            "top_k": TOP_K,
            "strategy": retriever.strategy,
            "generator": {
                "model": client.model,
                "digest": model_digest(host, client.model),
                "prompt": "fastapi_rag_lab.api.prompt.QUERY_SYSTEM_PROMPT",
            },
            "embedding_runtime": runtime,
            "provenance": collect_provenance(
                dataset_path=args.dataset, entry_count=len(entries), argv=command
            ),
        }
        run_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    done = {i["item_id"] for i in read_items(items_path)}
    for k, entry in enumerate(sample, start=1):
        item_id = f"j-{k:02d}"
        if item_id in done:
            continue
        item = make_item(item_id, entry, retriever, client)
        with items_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{item_id} {entry.id} {item['generation_s']}s", flush=True)
    print(f"{len(read_items(items_path))} items in {items_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
