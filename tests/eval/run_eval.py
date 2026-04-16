#!/usr/bin/env python
"""Run the full evaluation suite.

Usage:
    OLLAMA_HOST=http://172.19.64.1:11434 \\
      uv run python tests/eval/run_eval.py

Writes a timestamped JSON report under tests/eval/results/ and prints a
summary table to stdout. Exits 0 if every threshold in
fastapi_rag_lab.eval.runner.THRESHOLDS is met, 1 otherwise.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi_rag_lab.eval.runner import print_summary, run_eval


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    dataset_path = Path(__file__).parent / "golden_dataset.json"
    output_dir = Path(__file__).parent / "results"

    results = run_eval(dataset_path=dataset_path, output_dir=output_dir)
    print_summary(results)

    failed = [m for m, ok in results.threshold_status.items() if not ok]
    if failed:
        print(f"FAILED thresholds: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
