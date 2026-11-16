"""Agreement over both calibration batches (n = 100): the two human labellers
and Sonnet v1, plus the paired bootstrap of the gap between them.

Combines the 2026-10-30 dev batch and the 2026-11-10 held-out test batch
(item ids prefixed ``dev:`` / ``test:``) and reuses the package's metrics and
resampling (``agreement.compare``: Cohen's kappa, percentile bootstrap,
B = 10,000, seed 0). The paired difference resamples the same items for both
kappas, so it is the right test of "the judge agrees with me less than a
second person does".

    uv run python reports/2026-11-16-judge-agreement-n100/combined.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from fastapi_rag_lab.eval.judge.agreement import (
    RESAMPLES,
    SEED,
    _percentile,
    cohens_kappa,
    compare,
)
from fastapi_rag_lab.eval.judge.judge import LABELS

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "agreement.json"
BATCHES = {
    "dev": (
        "reports/2026-10-30-judge-items-so-dev-50",
        "reports/2026-10-30-judge-sonnet-so-dev-50",
    ),
    "test": (
        "reports/2026-11-10-judge-items-so-test-50",
        "reports/2026-11-16-judge-sonnet-v1-so-test-50",
    ),
}


def read_csv(path: Path, prefix: str) -> dict[str, str]:
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
    human: dict[str, str], second: dict[str, str], judge: dict[str, str]
) -> dict[str, object]:
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
    value = cohens_kappa(h, s) - cohens_kappa(h, j)
    rng = random.Random(SEED)
    diffs: list[float] = []
    for _ in range(RESAMPLES):
        pick = rng.choices(range(n), k=n)
        ks = cohens_kappa([h[i] for i in pick], [s[i] for i in pick])
        kj = cohens_kappa([h[i] for i in pick], [j[i] for i in pick])
        if ks is not None and kj is not None:
            diffs.append(ks - kj)
    diffs.sort()
    return {
        "n": n,
        "value": value,
        "ci_low": _percentile(diffs, 0.025),
        "ci_high": _percentile(diffs, 0.975),
        "share_le_zero": sum(d <= 0 for d in diffs) / len(diffs),
    }


def main() -> int:
    human: dict[str, str] = {}
    second: dict[str, str] = {}
    sonnet: dict[str, str] = {}
    for prefix, (items, run) in BATCHES.items():
        human |= read_csv(ROOT / items / "labels.csv", prefix)
        second |= read_csv(ROOT / items / "labels_second.csv", prefix)
        sonnet |= read_judgements(ROOT / run / "judgements.jsonl", prefix)
    comparisons = {
        "human_vs_second": compare(human, second),
        "human_vs_sonnet_v1": compare(human, sonnet),
        "second_vs_sonnet_v1": compare(second, sonnet),
    }
    report = {
        "batches": {k: list(v) for k, v in BATCHES.items()},
        "comparisons": comparisons,
        "paired_difference_human_second_minus_human_sonnet_v1": paired_difference(
            human, second, sonnet
        ),
        "kappa_ci_width": {
            "human_vs_second": {
                "dev_only": None,
                "combined": comparisons["human_vs_second"]["kappa"]["ci_high"]
                - comparisons["human_vs_second"]["kappa"]["ci_low"],
            }
        },
        "bootstrap": {
            "B": RESAMPLES,
            "seed": SEED,
            "confidence": 0.95,
            "method": "percentile bootstrap, items resampled with replacement",
        },
    }
    dev_report = ROOT / "reports/2026-11-10-judge-agreement-two-humans/agreement.json"
    dev = json.loads(dev_report.read_text())["comparisons"]["human_vs_second"]["kappa"]
    report["kappa_ci_width"]["human_vs_second"]["dev_only"] = (
        dev["ci_high"] - dev["ci_low"]
    )
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def ci(s: dict[str, object]) -> str:
        return f"{s['value']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"

    for name, c in comparisons.items():
        print(f"{name:22s} n={c['n']:3d} kappa {ci(c['kappa'])}")
    d = report["paired_difference_human_second_minus_human_sonnet_v1"]
    print(f"paired difference      n={d['n']:3d} {ci(d)}")
    print(f"share of resamples with difference <= 0: {d['share_le_zero']:.3f}")
    w = report["kappa_ci_width"]["human_vs_second"]
    print(f"human_vs_second kappa CI width: {w['dev_only']:.3f} -> {w['combined']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
