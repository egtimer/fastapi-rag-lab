# Stack Overflow label precision check

`labels.csv` lists every label of the Stack Overflow set as a (question, docs
page) pair: 578 pairs from 364 questions, in a random order fixed by seed 0.
The rows with `in_sample = yes` (ranks 1-60) are the sample to review. Review
them in rank order; if more precision is wanted later, continue down the list
(the first 60 rows do not change).

Made with

```bash
uv run python -m fastapi_rag_lab.eval.label_audit \
  --questions tests/eval/stackoverflow/questions.jsonl \
  --out tests/eval/stackoverflow/label_audit --n 60 --seed 0
```

## What to check

For each row, open the question (`question_link`), the answers that linked the
docs (`answers_with_docs_links`) and the page as indexed
(`page_at_pinned_commit`, the corpus commit the benchmarks use). Then write one
of these in `verdict`:

- `relevant`: the page answers the question, or explains the thing the
  question is stuck on; someone given only this page would make real progress.
- `background`: the page is about the right topic but does not help with this
  question's problem (general reading, a link mentioned in passing).
- `wrong`: the page is not about the question (a misresolved link, an
  unrelated page).

Use `note` for anything that does not fit, for example when the question
itself is not about FastAPI.

## How it will be used

Label precision is the share of reviewed pairs marked `relevant`, with a 95%
confidence interval; `relevant` + `background` gives a lenient version. With
60 pairs the interval is roughly ±10 points, so it says whether the labels are
mostly right, not their exact precision. The reviewer is the only labeler; the
verdicts are not used to change the benchmark labels.
