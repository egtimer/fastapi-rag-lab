# ADR 009: A Faithfulness Judge, Calibrated Against Human Labels

## Status
Accepted -- 2026-10-30. Addendum 2026-11-16: second labeller, stability and a
pre-registered held-out 50 (see the end of this ADR).

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

Judges: `reports/2026-10-30-judge-haiku-so-dev-50` and
`reports/2026-10-30-judge-sonnet-so-dev-50`; agreement:
`reports/2026-10-30-judge-agreement-so-dev-50/agreement.json`, from

```bash
uv run python -m fastapi_rag_lab.eval.judge.agreement \
  --items reports/2026-10-30-judge-items-so-dev-50 \
  --judge haiku=reports/2026-10-30-judge-haiku-so-dev-50 \
  --judge sonnet=reports/2026-10-30-judge-sonnet-so-dev-50 \
  --out reports/2026-10-30-judge-agreement-so-dev-50/agreement.json
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

## Addendum (2026-11-16): second labeller, stability, held-out 50

The four open points of this ADR were run as described in
[`docs/judge-followups.md`](../judge-followups.md). The decision above stands:
Sonnet 5.5 with prompt v1 at effort `low` is the judge. Write-up:
[Measure the judge before you trust it](https://egtimer.dev/en/blog/measure-the-judge/).

**Second labeller (dev 50).** A second engineer labelled the same 50 answers
blind, without AI assistance (`labels_second.csv` in
`reports/2026-10-30-judge-items-so-dev-50`): kappa 0.87 [0.71, 1.00] with the
first labeller (n = 49, one `unsure`); Sonnet 0.70 [0.47, 0.88] and Haiku 0.60
[0.35, 0.82] with the second labeller
(`reports/2026-11-10-judge-agreement-two-humans`).

**Stability (dev 50).** Sonnet v1 at effort `low` run six times
(`reports/2026-11-16-judge-agreement-retest-x6`): kappa with the first
labeller 0.82, 0.68, 0.82, 0.77, 0.82, 0.82 (mean 0.79, SD 0.055); run-to-run
kappa 0.86 to 1.00; 45 of 50 answers get the same verdict in all six runs. The
0.82 reported above was the top of this range.

**Explorations (dev 50, not results).** Chosen after the labels were read, so
they cannot show an improvement on these items
(`reports/2026-11-10-judge-agreement-retest`): effort `medium` 0.86
[0.68, 1.00]; Haiku with forced structured output, no unparsed replies but
kappa 0.59 [0.36, 0.80], with more false `supported`.

**Held-out 50.** New items from the Stack Overflow test split (seed 1,
`reports/2026-11-10-judge-items-so-test-50`). Items, labelling sheets and the
sha256 of prompts v1 and v2 were committed before labelling (af60c2d); the
second labeller's labels and hash in f43cd86, the first labeller's in 48fde44,
both before the first judge run on these items (2026-11-16 09:31 UTC). v2 is
v1 plus the citation rule (invented quotations and claims attributed to a
source that does not make them are `unsupported`).

| Comparison (held-out, n = 50) | Accuracy | Cohen's kappa |
|---|---|---|
| First vs second labeller | 0.96 [0.90, 1.00] | 0.92 [0.78, 1.00] |
| First labeller vs Sonnet v1 | 0.88 [0.78, 0.96] | 0.74 [0.53, 0.91] |
| First labeller vs Sonnet v2 | 0.90 [0.82, 0.98] | 0.78 [0.59, 0.95] |
| First labeller vs Sonnet v1, effort `medium` | 0.88 | 0.74, same verdicts as v1 on all 50 |

All judge disagreements on these items are answers the first labeller called
`supported` (6 with v1, 5 with v2). v2 changed one verdict, which is within
the run-to-run noise, so v1 is kept; effort `medium` changed none.

**Both batches (n = 100).** `reports/2026-11-16-judge-agreement-n100`
(`uv run python -m fastapi_rag_lab.eval.judge.combine`; `combined.py` in that
directory runs the same command):

| Comparison | n | Accuracy | Cohen's kappa |
|---|---|---|---|
| First vs second labeller | 99 | 0.95 [0.91, 0.99] | 0.89 [0.79, 0.98] |
| First labeller vs Sonnet v1 | 100 | 0.90 [0.84, 0.95] | 0.78 [0.64, 0.90] |
| Second labeller vs Sonnet v1 | 99 | 0.87 [0.80, 0.93] | 0.71 [0.56, 0.85] |

Paired over the same 99 items, human–human kappa minus first-labeller–judge
kappa is +0.12 [−0.04, +0.28] (6% of resamples at or below zero). Confusion
with the first labeller (rows human, columns judge; `supported`,
`unsupported`): [[29, 8], [2, 61]]; the judge is the stricter reader.

**Consequences.**

- The judge's agreement is reported as 0.78 [0.64, 0.90] on n = 100, not 0.82
  on n = 50.
- A single run's kappa carries judge noise (SD 0.055 over six runs) on top of
  the sampling interval; claims about a prompt or setting change smaller than
  that need repeated runs.
- Judging the 251 golden and Stack Overflow test questions is still not run.
