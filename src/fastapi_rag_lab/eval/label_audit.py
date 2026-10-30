"""Sample Stack Overflow (question, docs page) labels for a human precision check.

Every label is a pair: a question and one docs page that an answerer linked.
All pairs are shuffled with a fixed seed and written to a CSV in that order;
the first ``--n`` rows are the sample to review, and later rows extend it
without changing it. The reviewer fills ``verdict`` (see the README written
next to the CSV); nothing here labels anything.

    uv run python -m fastapi_rag_lab.eval.label_audit \\
        --questions tests/eval/stackoverflow/questions.jsonl \\
        --out tests/eval/stackoverflow/label_audit --n 60 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from fastapi_rag_lab.ingest.fetcher import PINNED_COMMIT_SHA

COLUMNS = (
    "rank",
    "in_sample",
    "pair_id",
    "split",
    "question",
    "question_link",
    "label_page",
    "page_at_pinned_commit",
    "answers_with_docs_links",
    "verdict",
    "note",
)
DOCS_BLOB = f"https://github.com/fastapi/fastapi/blob/{PINNED_COMMIT_SHA}/docs/en/docs/"


def load_pairs(questions_path: Path) -> list[dict]:
    pairs = []
    with questions_path.open(encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            for page in q["expected_sources"]:
                pairs.append(
                    {
                        "pair_id": f"{q['id']}:{page}",
                        "split": q["split"],
                        "question": q["query"],
                        "question_link": q["link"],
                        "label_page": page,
                        "page_at_pinned_commit": DOCS_BLOB + page,
                        "answers_with_docs_links": " ".join(
                            f"https://stackoverflow.com/a/{a}"
                            for a in q["label_answer_ids"]
                        ),
                    }
                )
    return pairs


def shuffled(pairs: list[dict], seed: int) -> list[dict]:
    order = sorted(pairs, key=lambda p: p["pair_id"])
    random.Random(seed).shuffle(order)
    return order


def write_csv(pairs: list[dict], path: Path, n: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for rank, pair in enumerate(pairs, start=1):
            writer.writerow(
                {
                    **pair,
                    "rank": rank,
                    "in_sample": "yes" if rank <= n else "",
                    "verdict": "",
                    "note": "",
                }
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    pairs = shuffled(load_pairs(args.questions), args.seed)
    if not 0 < args.n <= len(pairs):
        print(f"error: --n must be between 1 and {len(pairs)}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(pairs, args.out / "labels.csv", args.n)
    print(f"{len(pairs)} pairs, first {args.n} marked; seed {args.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
