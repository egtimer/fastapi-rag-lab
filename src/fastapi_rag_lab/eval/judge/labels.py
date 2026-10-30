"""Labelling sheet for the human calibration labels, and reading them back.

``sheet`` writes, into the items run, ``labeling_sheet_NN.md`` files of
``PER_FILE`` items each (the question, the sources the generator saw and its
answer; never a judge verdict; split so GitHub can render them) and
``labels.csv`` with an empty ``label`` column. The labeller fills in
``supported``, ``unsupported`` or ``unsure`` per item, by the same definitions
the judge gets (written at the top of the sheet).

    uv run python -m fastapi_rag_lab.eval.judge.labels \\
        --items reports/2026-10-30-judge-items-so-dev-50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from fastapi_rag_lab.eval.judge.answers import read_items
from fastapi_rag_lab.eval.judge.judge import LABELS

UNSURE = "unsure"
HUMAN_LABELS = (*LABELS, UNSURE)
DEFAULT_QUESTIONS = Path("tests/eval/stackoverflow/questions.jsonl")
PER_FILE = 10

INSTRUCTIONS = """\
# Judge calibration: labelling sheet

For each item, read the question, the sources the model was given and its
answer, then write one label per item in `labels.csv` (same folder):

- `supported`: every factual claim in the answer (what FastAPI does, how an
  API or parameter behaves, code presented as working) is stated in the
  sources or follows directly from them. Rewording, summarising and generic
  connecting text are fine. Saying the sources do not cover the question is
  supported when they indeed do not.
- `unsupported`: at least one claim is not in the sources or contradicts them,
  including code or parameter names the sources do not show, and claiming the
  sources lack something they contain.
- `unsure`: you cannot decide; these items are left out of the agreement.

Judge support only, not whether the answer is helpful or complete. Use what is
in the sources, not what you know about FastAPI. `note` is optional. The
judge's verdicts are not in this folder; do not look at judge runs before
finishing.
"""


def question_links(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {q["id"]: q["link"] for q in map(json.loads, fh) if q.get("link")}


def render_item(item: dict[str, Any], link: str | None) -> str:
    title = item["question"].strip()
    lines = [f"## {item['item_id']}: {title}", ""]
    if link:
        lines += [f"Question: [{item['question_id']}]({link}) (CC BY-SA 4.0)", ""]
    for i, src in enumerate(item["sources"], 1):
        where = " > ".join(src.get("heading_path") or []) or src.get("source_file", "")
        lines += [
            "<details>",
            f"<summary>Source {i}: {src.get('source_file', '')} ({where})</summary>",
            "",
            src["parent_text"].strip(),
            "",
            "</details>",
            "",
        ]
    lines += ["**Answer:**", "", item["answer"].strip() or "(empty)", "", "---", ""]
    return "\n".join(lines)


def write_sheet(
    items: list[dict[str, Any]], run_dir: Path, links: dict[str, str]
) -> None:
    for k in range(0, len(items), PER_FILE):
        chunk = items[k : k + PER_FILE]
        body = [INSTRUCTIONS, ""] + [
            render_item(it, links.get(it["question_id"])) for it in chunk
        ]
        name = f"labeling_sheet_{k // PER_FILE + 1:02d}.md"
        (run_dir / name).write_text("\n".join(body), encoding="utf-8")
    csv_path = run_dir / "labels.csv"
    if csv_path.exists():
        raise FileExistsError(f"{csv_path} exists; not overwriting labels")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item_id", "label", "note"])
        for it in items:
            writer.writerow([it["item_id"], "", ""])


def read_labels(path: Path, item_ids: set[str]) -> dict[str, str]:
    """Labels by item id; raises on an unknown id, a bad label or a missing one."""
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    labels: dict[str, str] = {}
    for row in rows:
        item_id, label = row["item_id"].strip(), row["label"].strip().lower()
        if item_id not in item_ids:
            raise ValueError(f"unknown item id {item_id!r}")
        if label not in HUMAN_LABELS:
            raise ValueError(f"{item_id}: label {label!r} not in {HUMAN_LABELS}")
        labels[item_id] = label
    missing = sorted(item_ids - labels.keys())
    if missing:
        raise ValueError(f"no label for {len(missing)} items: {missing[:5]}")
    return labels


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--items", type=Path, required=True, help="items run directory")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = p.parse_args(argv)
    items = read_items(args.items / "items.jsonl")
    if not items:
        print(f"error: no items in {args.items}", file=sys.stderr)
        return 2
    try:
        write_sheet(items, args.items, question_links(args.questions))
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote the labelling sheets and labels.csv for {len(items)} items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
