"""Writes agreement.json in this directory: the two human labellers and Sonnet v1
over both calibration batches (n = 100), plus the paired bootstrap of the gap.

The code is ``fastapi_rag_lab.eval.judge.combine`` (tested in
``tests/eval/test_combine.py``); this file keeps the command this report was
made with working:

    uv run python reports/2026-11-16-judge-agreement-n100/combined.py
"""

from __future__ import annotations

from fastapi_rag_lab.eval.judge.combine import main

if __name__ == "__main__":
    raise SystemExit(main([]))
