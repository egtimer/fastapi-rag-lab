# Gate: Haiku with the schema forced, against Haiku without it

The judge-change gate (`fastapi_rag_lab.eval.judge.gate`) run on the committed runs of
[`2026-11-18-judge-agreement-haiku-schema-x3`](../2026-11-18-judge-agreement-haiku-schema-x3/):
the baseline is the four plain Haiku runs, the candidate the four runs with the output schema
forced. No API call; the inputs are the committed `judgements.jsonl` files.

```bash
uv run python -m fastapi_rag_lab.eval.judge.gate \
    --items reports/2026-10-30-judge-items-so-dev-50 \
    --baseline reports/2026-10-30-judge-haiku-so-dev-50 \
    --baseline reports/2026-11-18-judge-haiku-plain-r1-so-dev-50 \
    --baseline reports/2026-11-18-judge-haiku-plain-r2-so-dev-50 \
    --baseline reports/2026-11-18-judge-haiku-plain-r3-so-dev-50 \
    --candidate reports/2026-11-10-judge-haiku-structured-so-dev-50 \
    --candidate reports/2026-11-18-judge-haiku-structured-r1-so-dev-50 \
    --candidate reports/2026-11-18-judge-haiku-structured-r2-so-dev-50 \
    --candidate reports/2026-11-18-judge-haiku-structured-r3-so-dev-50 \
    --out reports/2026-11-19-judge-gate-haiku-structured/gate.json
```

Items compared: the 28 answers I labelled `unsupported` that all eight runs gave a verdict for.

| | False passes per run (of 28) | |
|---|---|---|
| Baseline (plain) | 4, 6, 3, 4 | range 11% to 21% |
| Candidate (schema forced) | 8, 9, 7, 6 | mean 27% |

**Result: FAIL** (exit code 1). The candidate's mean, 27%, is above the highest baseline run,
21%. Forcing the schema would not have been accepted as a judge change, even though it brought
unparsed replies from 2–3 per run to zero. One candidate run alone (6 of 28) sits at the
ceiling and would pass; that is why the gate takes the mean of several candidate runs when they
exist.

`tests/eval/test_gate.py` rebuilds `gate.json` from the committed runs and checks it matches,
and checks that three Sonnet re-runs of the reported judge pass against the other three.
