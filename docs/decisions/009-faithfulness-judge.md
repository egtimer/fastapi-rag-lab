# ADR 009: A Faithfulness Judge, Calibrated Against Human Labels

## Status
Accepted -- 2026-10-31

## Context

The retrieval benchmarks (ADR 006, 008) say whether the right docs pages come
back; they say nothing about the answer `gemma3:4b` writes from them. The
RAGAS pipeline (ADR 005) uses the same small model as its own judge and has no
committed results.

The Stack Overflow set labels docs pages, not answers, so there is no reference
answer to compare with. What can be checked without one is faithfulness: does
the answer only say what its sources say? That is the hallucination question
this project cares about, and an LLM judge can answer it cheaply if it agrees
with a human. This ADR measures that agreement before the judge is used for
any reported number.

## Decision

**Judge.** One Claude call per answer, given the question, the five sources the
generator saw (in its order) and the answer; it returns
`{"rationale", "label"}` with `label` `supported` or `unsupported`. The prompt
is `JUDGE_SYSTEM` in `src/fastapi_rag_lab/eval/judge/judge.py`. A reply that
is not valid JSON with one of the two labels is `unparsed` and is left out of
the agreement, not guessed. Effort `low`, as in agentic-sql-assistant, whose
judge code this adapts.

**Labelling rule** (the human labeller's, and the definitions in the labelling
sheet):

- `supported` means faithful to the five sources, not a useful answer.
- Any material claim that is not in the sources is `unsupported`, even if it
  is true of FastAPI.
- An invented quotation, or attributing to a source something it does not
  say, is `unsupported`.
- Saying that the sources do not cover the question is `supported` when they
  indeed do not.

The judge's prompt states the first, second and fourth points; the citation
rule is not explicit in it. The prompt was frozen before the human labels
were read and was not changed afterwards.

**Calibration set.** 50 questions drawn at random (seed 0) from the Stack
Overflow dev split, answered through the API's own path: `hybrid` retrieval,
top_k 5, the API prompt, `gemma3:4b` on Ollama 0.35.1
(`reports/2026-10-30-judge-items-so-dev-50`, with the run's provenance). One
person labelled all 50 blind to the judges, from the labelling sheets in that
directory: 17 `supported`, 33 `unsupported`, no `unsure`.

## Results

Judges: `reports/2026-10-31-judge-haiku-so-dev-50` and
`reports/2026-10-31-judge-sonnet-so-dev-50`; agreement:
`reports/2026-10-31-judge-agreement-so-dev-50/agreement.json`, from

```bash
uv run python -m fastapi_rag_lab.eval.judge.agreement \
  --items reports/2026-10-30-judge-items-so-dev-50 \
  --judge haiku=reports/2026-10-31-judge-haiku-so-dev-50 \
  --judge sonnet=reports/2026-10-31-judge-sonnet-so-dev-50 \
  --out reports/2026-10-31-judge-agreement-so-dev-50/agreement.json
```

95% percentile bootstrap CIs, items resampled, B = 10,000, seed 0. Per-class
agreement is the share of the human's `supported` (or `unsupported`) items to
which the judge gave the same label.

| Judge | n | Accuracy | Cohen's kappa | On human `supported` | On human `unsupported` | Unparsed | Cost per answer |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 47 | 0.87 [0.77, 0.96] | 0.73 [0.51, 0.91] | 0.88 [0.70, 1.00] (n = 17) | 0.87 [0.73, 0.97] (n = 30) | 3 | $0.00054 |
| Claude Sonnet 5.5 | 50 | 0.92 [0.84, 0.98] | 0.82 [0.63, 0.96] | 0.88 [0.70, 1.00] (n = 17) | 0.94 [0.85, 1.00] (n = 33) | 0 | $0.0089 |

- Haiku's three unparsed replies were JSON with unescaped quotes inside the
  rationale; the human had labelled all three `unsupported`, so leaving them
  out flatters Haiku. Counting them as disagreements, Haiku matches the human
  on 41 of 50 answers (0.82).
- Confusion matrices (rows human, columns judge; `supported`, `unsupported`):
  Haiku [[15, 2], [4, 26]], Sonnet [[15, 2], [2, 31]].
- The two judges agree with each other on 41 of 47 answers (kappa 0.73
  [0.51, 0.91]).
- The calibration cost $0.027 (Haiku) and $0.443 (Sonnet); judge prompts
  averaged 3,795 input tokens.

By the human labels, 17 of the 50 answers (34%) are fully supported by their
sources. This is a statement about these 50 dev questions, labelled by one
person; it is not yet a measured hallucination rate for the system.

## Consequences

- Both judges agree with the human well beyond chance; their intervals
  overlap, so n = 50 does not separate them. Sonnet had no unparsed replies
  and the higher point estimates; Haiku costs about 16 times less per answer.
- The judge prompt stays as it was frozen. Changing it now, after reading
  these 50 labels, and re-scoring it on the same 50 would tune the judge to
  its own evaluation set.
- Judging the 251 questions of golden + Stack Overflow test is not run in this
  phase. At the measured cost per answer it would cost about $0.14 with Haiku
  or $2.22 with Sonnet, plus about 10 hours of `gemma3:4b` generation on the
  4-core CPU; which judge to use is decided when it is run.
- One labeller and 50 items: the CIs are wide, and the labels encode one
  person's reading of "material claim".

## Future work

Each of these changes the judge, so its agreement must be measured again on a
new set of human-labelled answers, not on these 50:

- Make the citation rule explicit in the judge prompt (invented quotations and
  claims attributed to a source that does not make them are `unsupported`).
- Force structured output for Haiku (for example a JSON schema on the
  response) so that malformed JSON can no longer leave answers unparsed.
