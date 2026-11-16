"""Agreement over several calibration batches pooled together, plus the paired
bootstrap of the gap between human-human and human-judge kappa.

Each batch is an items run (with ``labels.csv`` and ``labels_second.csv``) and
a judge run of the same items. Item ids are prefixed with the batch name
(``dev:j-01``) so batches can share ids. Metrics and resampling are those of
``agreement.compare`` (Cohen's kappa, percentile bootstrap, B = 10,000, seed 0).
The paired difference resamples the same items for both kappas, so it tests
"the judge agrees with the first labeller less than a second person does".

With no arguments it rebuilds ``reports/2026-11-16-judge-agreement-n100``
(the batches in ``N100_BATCHES``):

    uv run python -m fastapi_rag_lab.eval.judge.combine

Other batches:

    uv run python -m fastapi_rag_lab.eval.judge.combine \\
        --batch dev=<items run>,<judge run> --batch test=<items run>,<judge run> \\
        --dev-report <agreement.json of one batch> --out <dir>/agreement.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi_rag_lab.eval.judge.agreement import (
    RESAMPLES,
    SEED,
    cohens_kappa,
    compare,
    percentile,
)
from fastapi_rag_lab.eval.judge.judge import LABELS

if TYPE_CHECKING:
    from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[4]

N100_BATCHES: dict[str, tuple[str, str]] = {
    "dev": (
        "reports/2026-10-30-judge-items-so-dev-50",
        "reports/2026-10-30-judge-sonnet-so-dev-50",
    ),
    "test": (
        "reports/2026-11-10-judge-items-so-test-50",
        "reports/2026-11-16-judge-sonnet-v1-so-test-50",
    ),
}
N100_DEV_REPORT = "reports/2026-11-10-judge-agreement-two-humans/agreement.json"
N100_OUT = "reports/2026-11-16-judge-agreement-n100/agreement.json"


def read_labels_csv(path: Path, prefix: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {
            f"{prefix}:{r['item_id'].strip()}": r["label"].strip()
            for r in csv.DictReader(fh)
        }


def read_judgements(path: Path, prefix: str) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[f"{prefix}:{d['item_id']}"] = d["label"]
    return out


def paired_difference(
    human: Mapping[str, str],
    second: Mapping[str, str],
    judge: Mapping[str, str],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """kappa(human, second) - kappa(human, judge) on the items all three
    labelled, resampled together."""
    ids = sorted(
        i
        for i in human.keys() & second.keys() & judge.keys()
        if human[i] in LABELS and second[i] in LABELS and judge[i] in LABELS
    )
    h = [human[i] for i in ids]
    s = [second[i] for i in ids]
    j = [judge[i] for i in ids]
    n = len(ids)
    ks0, kj0 = cohens_kappa(h, s), cohens_kappa(h, j)
    value = None if ks0 is None or kj0 is None else ks0 - kj0
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(resamples):
        pick = rng.choices(range(n), k=n)
        ks = cohens_kappa([h[i] for i in pick], [s[i] for i in pick])
        kj = cohens_kappa([h[i] for i in pick], [j[i] for i in pick])
        if ks is not None and kj is not None:
            diffs.append(ks - kj)
    diffs.sort()
    return {
        "n": n,
        "value": value,
        "ci_low": percentile(diffs, 0.025) if diffs else None,
        "ci_high": percentile(diffs, 0.975) if diffs else None,
        "share_le_zero": sum(d <= 0 for d in diffs) / len(diffs) if diffs else None,
    }


def combined_report(
    root: Path,
    batches: Mapping[str, tuple[str, str]],
    dev_report: str | None = None,
) -> dict[str, Any]:
    """Pool the batches and compare the first labeller, the second labeller and
    the judge. ``dev_report`` (an agreement.json of one batch alone) adds the
    width of its human-human kappa CI, to show how much pooling narrowed it."""
    human: dict[str, str] = {}
    second: dict[str, str] = {}
    judge: dict[str, str] = {}
    for prefix, (items, run) in batches.items():
        human |= read_labels_csv(root / items / "labels.csv", prefix)
        second |= read_labels_csv(root / items / "labels_second.csv", prefix)
        judge |= read_judgements(root / run / "judgements.jsonl", prefix)
    comparisons = {
        "human_vs_second": compare(human, second),
        "human_vs_sonnet_v1": compare(human, judge),
        "second_vs_sonnet_v1": compare(second, judge),
    }
    hs = comparisons["human_vs_second"]["kappa"]
    dev_only = None
    if dev_report is not None:
        dev = json.loads((root / dev_report).read_text(encoding="utf-8"))
        dk = dev["comparisons"]["human_vs_second"]["kappa"]
        dev_only = dk["ci_high"] - dk["ci_low"]
    return {
        "batches": {k: list(v) for k, v in batches.items()},
        "comparisons": comparisons,
        "paired_difference_human_second_minus_human_sonnet_v1": paired_difference(
            human, second, judge
        ),
        "kappa_ci_width": {
            "human_vs_second": {
                "dev_only": dev_only,
                "combined": hs["ci_high"] - hs["ci_low"],
            }
        },
        "bootstrap": {
            "B": RESAMPLES,
            "seed": SEED,
            "confidence": 0.95,
            "method": "percentile bootstrap, items resampled with replacement",
        },
    }


def format_report(report: Mapping[str, Any]) -> str:
    def ci(s: Mapping[str, Any]) -> str:
        return f"{s['value']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"

    lines = [
        f"{name:22s} n={c['n']:3d} kappa {ci(c['kappa'])}"
        for name, c in report["comparisons"].items()
    ]
    d = report["paired_difference_human_second_minus_human_sonnet_v1"]
    lines.append(f"paired difference      n={d['n']:3d} {ci(d)}")
    lines.append(f"share of resamples with difference <= 0: {d['share_le_zero']:.3f}")
    w = report["kappa_ci_width"]["human_vs_second"]
    if w["dev_only"] is not None:
        lines.append(
            "human_vs_second kappa CI width: "
            f"{w['dev_only']:.3f} -> {w['combined']:.3f}"
        )
    return "\n".join(lines)


def _batch_arg(value: str) -> tuple[str, tuple[str, str]]:
    name, sep, rest = value.partition("=")
    items, comma, run = rest.partition(",")
    if not sep or not name or not comma or not items or not run:
        raise argparse.ArgumentTypeError(
            f"expected NAME=ITEMS_RUN,JUDGE_RUN, got {value!r}"
        )
    return name, (items, run)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--batch",
        type=_batch_arg,
        action="append",
        help="NAME=ITEMS_RUN,JUDGE_RUN (repeatable; default: the n = 100 batches)",
    )
    p.add_argument(
        "--dev-report",
        help="agreement.json of one batch alone, for the CI width comparison",
    )
    p.add_argument("--out", type=Path, help=f"default: {N100_OUT}")
    p.add_argument(
        "--root", type=Path, default=ROOT, help="repository root (default: this repo)"
    )
    args = p.parse_args(argv)
    if args.batch:
        batches, dev_report = dict(args.batch), args.dev_report
    else:
        batches, dev_report = N100_BATCHES, args.dev_report or N100_DEV_REPORT
    out = args.out or args.root / N100_OUT
    report = combined_report(args.root, batches, dev_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
