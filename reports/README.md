# Reports

Each subdirectory is one committed benchmark run: the raw `benchmark.json` written by `tests/eval/run_benchmarks.py`, plus an `analysis.json` produced from it by `python -m fastapi_rag_lab.eval.analyze` (bootstrap 95% confidence intervals and paired comparisons between strategies). Numbers quoted in the top-level README should come from these files, so they can be checked and re-derived without re-running the benchmark. The analysis records the input file's sha256, the seed and the command used.
