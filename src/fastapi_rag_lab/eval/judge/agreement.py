"""Agreement between judges and the human labels: accuracy, Cohen's kappa, confusion.

Each comparison uses the items both raters labelled with ``supported`` or
``unsupported`` (human ``unsure`` and judge ``unparsed``/``error`` are left out
and counted). Every number has a 95% percentile bootstrap CI (items resampled
with replacement, B = 10000, seed 0). A resample where kappa is undefined (both
raters used one label only) is left out of the interval and counted in
``frac_undefined``. Adapted from agentic-sql-assistant
(``evals/judge/agreement.py``).

    uv run python -m fastapi_rag_lab.eval.judge.agreement \\
        --items reports/2026-10-30-judge-items-so-dev-50 \\
        --judge haiku=reports/<haiku run> --judge sonnet=reports/<sonnet run> \\
        --out reports/<agreement run>/agreement.json

A second human labeller (``--human NAME=labels_NAME.csv``) is one more rater:
``human_vs_NAME`` is the human-human agreement, the ceiling a judge is measured
against, and every judge is also compared with that labeller. Two runs of the
same judge passed as two ``--judge`` entries give its test-retest agreement.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi_rag_lab.eval.judge.items import read_items
from fastapi_rag_lab.eval.judge.judge import LABELS, SUPPORTED
from fastapi_rag_lab.eval.judge.labels import read_labels

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    Metric = Callable[[Sequence[str], Sequence[str]], float | None]

RESAMPLES = 10_000
SEED = 0


def accuracy(a: Sequence[str], b: Sequence[str]) -> float | None:
    if not a:
        return None
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """``(po - pe) / (1 - pe)``; None when ``pe`` is 1 (one shared label only)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def class_agreement(label: str) -> Metric:
    """Share of the items ``a`` labels ``label`` that ``b`` labels the same.

    With the human as ``a`` this is the judge's recall on that class; None when
    ``a`` never uses the label.
    """

    def metric(a: Sequence[str], b: Sequence[str]) -> float | None:
        idx = [i for i, x in enumerate(a) if x == label]
        if not idx:
            return None
        return sum(b[i] == label for i in idx) / len(idx)

    return metric


def _percentile(sorted_values: list[float], q: float) -> float:
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap(
    a: Sequence[str], b: Sequence[str], metric: Metric, *, resamples: int, seed: int
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "value": metric(a, b),
        "ci_low": None,
        "ci_high": None,
        "frac_undefined": None,
    }
    n = len(a)
    if n == 0:
        return out
    rng = random.Random(seed)
    stats: list[float] = []
    undefined = 0
    for _ in range(resamples):
        pick = rng.choices(range(n), k=n)
        s = metric([a[i] for i in pick], [b[i] for i in pick])
        if s is None:
            undefined += 1
        else:
            stats.append(s)
    if stats:
        stats.sort()
        out["ci_low"], out["ci_high"] = (
            _percentile(stats, 0.025),
            _percentile(stats, 0.975),
        )
    out["frac_undefined"] = undefined / resamples
    return out


def compare(
    first: Mapping[str, str],
    second: Mapping[str, str],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    ids = sorted(
        i
        for i in first.keys() & second.keys()
        if first[i] in LABELS and second[i] in LABELS
    )
    a, b = [first[i] for i in ids], [second[i] for i in ids]
    pairs = Counter(zip(a, b, strict=True))
    return {
        "n": len(ids),
        "accuracy": bootstrap(a, b, accuracy, resamples=resamples, seed=seed),
        "kappa": bootstrap(a, b, cohens_kappa, resamples=resamples, seed=seed),
        "per_class_of_first": {
            label: {
                "n": a.count(label),
                **bootstrap(
                    a, b, class_agreement(label), resamples=resamples, seed=seed
                ),
            }
            for label in LABELS
        },
        "confusion": {
            "labels": list(LABELS),
            "rows": "first",
            "columns": "second",
            "matrix": [[pairs[(x, y)] for y in LABELS] for x in LABELS],
        },
        f"share_{SUPPORTED}": {
            "first": a.count(SUPPORTED) / len(a) if a else None,
            "second": b.count(SUPPORTED) / len(b) if b else None,
        },
    }


def agreement_report(
    human: Mapping[str, str],
    judges: Mapping[str, Mapping[str, str]],
    *,
    humans: Mapping[str, Mapping[str, str]] | None = None,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Pairwise comparisons: the human against every other rater, the other
    humans against every judge, and every pair of judges."""
    humans = dict(humans or {})
    raters = {"human": human, **humans, **judges}
    others = [*humans, *judges]
    pairs = [
        *(("human", r) for r in others),
        *((h, j) for h in humans for j in judges),
        *combinations(judges, 2),
    ]
    return {
        "counts": {
            name: dict(Counter(labels.values())) for name, labels in raters.items()
        },
        "comparisons": {
            f"{x}_vs_{y}": compare(raters[x], raters[y], resamples=resamples, seed=seed)
            for x, y in pairs
        },
        "bootstrap": {
            "B": resamples,
            "seed": seed,
            "confidence": 0.95,
            "method": "percentile bootstrap, items resampled with replacement",
        },
    }


def _ci(stat: Mapping[str, Any]) -> str:
    if stat["value"] is None:
        return "undefined"
    text = f"{stat['value']:.2f}"
    if stat["ci_low"] is not None:
        text += f" [{stat['ci_low']:.2f}, {stat['ci_high']:.2f}]"
    return text


def format_report(report: Mapping[str, Any]) -> str:
    lines = [f"labels: {report['counts']}", ""]
    lines.append(f"{'comparison':<20}{'n':>4}  {'accuracy [95% CI]':<24}kappa [95% CI]")
    for name, r in report["comparisons"].items():
        lines.append(
            f"{name:<20}{r['n']:>4}  {_ci(r['accuracy']):<24}{_ci(r['kappa'])}"
        )
    for name, r in report["comparisons"].items():
        per = r["per_class_of_first"]
        lines.append(
            f"{name} per class of first: "
            + ", ".join(f"{k} (n={v['n']}) {_ci(v)}" for k, v in per.items())
        )
    for name, r in report["comparisons"].items():
        lines.append(
            f"{name} confusion {r['confusion']['labels']}: {r['confusion']['matrix']}"
        )
    return "\n".join(lines)


def _judge_arg(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=RUN_DIR, got {value!r}")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--labels", type=Path, help="default: labels.csv in the items run")
    p.add_argument("--judge", type=_judge_arg, action="append", required=True)
    p.add_argument(
        "--human",
        type=_judge_arg,
        action="append",
        default=[],
        help="NAME=labels csv of another human labeller (repeatable)",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    ids = {i["item_id"] for i in read_items(args.items / "items.jsonl")}
    try:
        human = read_labels(args.labels or args.items / "labels.csv", ids)
        humans = {name: read_labels(path, ids) for name, path in args.human}
    except ValueError as exc:
        print(f"error: labels: {exc}", file=sys.stderr)
        return 2
    judges = {
        name: {j["item_id"]: j["label"] for j in read_items(path / "judgements.jsonl")}
        for name, path in args.judge
    }
    report = agreement_report(human, judges, humans=humans)
    report["items_run"] = args.items.name
    report["judge_runs"] = {name: path.name for name, path in args.judge}
    report["human_labels"] = {name: str(path) for name, path in args.human}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
