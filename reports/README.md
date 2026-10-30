# Reports

Each subdirectory is one committed benchmark run: the raw `benchmark.json` written by `tests/eval/run_benchmarks.py`, plus an `analysis.json` produced from it by `python -m fastapi_rag_lab.eval.analyze` (bootstrap 95% confidence intervals and paired comparisons between strategies). Numbers quoted in the top-level README should come from these files, so they can be checked and re-derived without re-running the benchmark. The analysis records the input file's sha256, the seed and the command used.

The judge runs `2026-10-30-judge-haiku-so-dev-50` and `2026-10-30-judge-sonnet-so-dev-50` were made on 2026-10-30 (see `started_at` in their `summary.json`) under the run ids `2026-10-31-judge-*-so-dev-50`, a mistyped date. The directories were renamed afterwards; `run_id` and `command` in their `summary.json` keep the ids as they were run. `2026-10-30-judge-agreement-so-dev-50/agreement.json` was regenerated with the renamed paths (same numbers).
