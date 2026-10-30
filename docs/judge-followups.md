# Judge calibration: the runs that follow ADR 009

ADR 009 measured two Claude judges against one human on 50 answers and left
four things open. This file is the runbook for them, in the order that
moves the result most. Costs are from the 2026-10-30 runs ($0.0089 per
answer for Sonnet, $0.00054 for Haiku, 3,795 input tokens per prompt).

Rules that apply to every run below:

- The v1 prompt stays frozen. v2 (`--prompt v2`) adds the citation rule the
  human applied; its agreement is measured only on NEW human labels, never on
  the 50 of ADR 009.
- Pre-register labels: hash a labels file and commit the hash BEFORE the judge
  run that will be compared with it, then pass it to the run.
- A judge run id is new every time; nothing is overwritten.

## 1. Second human labeller (the ceiling), $0

The sheets and an empty `labels_second.csv` are already in
`reports/2026-10-30-judge-items-so-dev-50/`, with `LABELLING_RULE.md` (the
sheet rule plus the citation rule the first labeller added mid-pass). The
second labeller reads the five `labeling_sheet_NN.md`, fills in
`labels_second.csv` and never opens a `judge-*` folder.

```bash
# before they start (already done for labels.csv: labels.csv.sha256)
uv run python -m fastapi_rag_lab.eval.judge.labels \
  --hash reports/2026-10-30-judge-items-so-dev-50/labels_second.csv   # when filled
git add reports/2026-10-30-judge-items-so-dev-50/labels_second.csv*
git commit -m "Second labeller: 50 dev answers"

uv run python -m fastapi_rag_lab.eval.judge.agreement \
  --items reports/2026-10-30-judge-items-so-dev-50 \
  --human second=reports/2026-10-30-judge-items-so-dev-50/labels_second.csv \
  --judge haiku=reports/2026-10-30-judge-haiku-so-dev-50 \
  --judge sonnet=reports/2026-10-30-judge-sonnet-so-dev-50 \
  --out reports/<date>-judge-agreement-two-humans/agreement.json
```

`human_vs_second` is the human-human kappa. Where the two humans disagree,
look at j-36 and j-42 first: that settles whether the first labeller or
Sonnet is the outlier on "follows directly from them".

## 2. Sonnet test-retest and effort `medium`, about $1, 4 minutes

Same prompt (v1), same 50 items. Three runs, three new ids.

```bash
export ANTHROPIC_API_KEY=...
uv run --extra judge python -m fastapi_rag_lab.eval.judge.run_judge \
  --items reports/2026-10-30-judge-items-so-dev-50 --model sonnet \
  --budget-usd 0.60 --run-id <date>-judge-sonnet-retest-so-dev-50 \
  --preregistered-labels reports/2026-10-30-judge-items-so-dev-50/labels.csv
uv run --extra judge python -m fastapi_rag_lab.eval.judge.run_judge \
  --items reports/2026-10-30-judge-items-so-dev-50 --model sonnet --effort medium \
  --budget-usd 0.80 --run-id <date>-judge-sonnet-medium-so-dev-50 \
  --preregistered-labels reports/2026-10-30-judge-items-so-dev-50/labels.csv
uv run --extra judge python -m fastapi_rag_lab.eval.judge.run_judge \
  --items reports/2026-10-30-judge-items-so-dev-50 --model haiku --structured \
  --budget-usd 0.10 --run-id <date>-judge-haiku-structured-so-dev-50 \
  --preregistered-labels reports/2026-10-30-judge-items-so-dev-50/labels.csv

uv run python -m fastapi_rag_lab.eval.judge.agreement \
  --items reports/2026-10-30-judge-items-so-dev-50 \
  --judge sonnet=reports/2026-10-30-judge-sonnet-so-dev-50 \
  --judge sonnet_retest=reports/<date>-judge-sonnet-retest-so-dev-50 \
  --judge sonnet_medium=reports/<date>-judge-sonnet-medium-so-dev-50 \
  --judge haiku_structured=reports/<date>-judge-haiku-structured-so-dev-50 \
  --out reports/<date>-judge-agreement-retest/agreement.json
```

`sonnet_vs_sonnet_retest` is test-retest. Check whether j-11, j-16, j-36 and
j-42 keep their labels across the three Sonnet runs. The structured Haiku
run is the v1 prompt with a forced tool call: it should have zero
`unparsed`, and its agreement tells whether the three unparsed replies of
2026-10-30 hid agreement or disagreement.

## 3. Prompt v2 on a fresh held-out 50 (closing the loop)

About 2 hours of `gemma3:4b` generation on the 4-core CPU, 2 to 3 hours of
labelling, $0.47 of judging. The 50 come from the Stack Overflow TEST split
with a new seed, so they overlap neither the dev calibration set nor each
other.

```bash
# answers (needs the index on Ollama 0.35.1, see answers.py)
uv run python -m fastapi_rag_lab.eval.judge.answers \
  --dataset tests/eval/stackoverflow/test.json --n 50 --seed 1 \
  --run-id <date>-judge-items-so-test-50
uv run python -m fastapi_rag_lab.eval.judge.labels --items reports/<date>-judge-items-so-test-50
# ... label blind, with LABELLING_RULE.md as the rule from the start ...
uv run python -m fastapi_rag_lab.eval.judge.labels --hash reports/<date>-judge-items-so-test-50/labels.csv
git add reports/<date>-judge-items-so-test-50 && git commit -m "Held-out 50: items and labels"

# v1 and v2 on the same new items, so the prompt change is the only difference
for v in v1 v2; do
uv run --extra judge python -m fastapi_rag_lab.eval.judge.run_judge \
  --items reports/<date>-judge-items-so-test-50 --model sonnet --prompt $v \
  --budget-usd 0.60 --run-id <date>-judge-sonnet-$v-so-test-50 \
  --preregistered-labels reports/<date>-judge-items-so-test-50/labels.csv
done
uv run python -m fastapi_rag_lab.eval.judge.agreement \
  --items reports/<date>-judge-items-so-test-50 \
  --judge sonnet_v1=reports/<date>-judge-sonnet-v1-so-test-50 \
  --judge sonnet_v2=reports/<date>-judge-sonnet-v2-so-test-50 \
  --out reports/<date>-judge-agreement-v2-so-test-50/agreement.json
```

The result to report is "v2 moved Sonnet from X/50 to Y/50 on new labels",
with the items that flipped. If v2 is not better on new labels, that is the
result; it goes in ADR 010 either way.

## 4. Tagging

After the runs above are committed, tag the commit and cite the tag in the
post instead of "the runs came from a working tree":

```bash
git tag -a judge-calibration-v1 -m "ADR 009 calibration, plumbing and follow-ups"
git push origin judge-calibration-v1
```
