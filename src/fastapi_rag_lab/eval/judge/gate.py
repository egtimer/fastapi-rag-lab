"""Gate for a change to the judge: its false-pass rate must stay in the earlier range.

A change to how the judge replies (the output schema, the prompt's reply format,
the user template, the model or its settings) can lower the parse-error rate and
still make the judge pass more answers the human labelled ``unsupported``. Forcing
the schema on Haiku did exactly that (``reports/2026-11-18-judge-agreement-haiku-
schema-x3``), and kappa alone hid how much. So a new judge configuration counts
only after this gate passes:

- the **baseline** is several runs of the current configuration, the
  **candidate** one or more runs of the new one, all on the same labelled items;
- the items compared are the ones the human labelled ``unsupported`` and every
  run, baseline and candidate, gave a verdict for (an unparsed or errored reply
  in any run removes the item for all, so each run is scored on the same
  answers);
- the gate passes when the candidate's mean false-pass rate is at most the
  highest rate among the baseline runs. More candidate runs give a steadier mean.

    uv run python -m fastapi_rag_lab.eval.judge.gate \\
        --items reports/2026-10-30-judge-items-so-dev-50 \\
        --baseline reports/<current run 1> --baseline reports/<current run 2> \\
        --candidate reports/<new run 1> --candidate reports/<new run 2> \\
        --out reports/<gate run>/gate.json

Exit code 0 when the gate passes, 1 when it fails, 2 on bad input. Needs no API
call: it reads committed judge runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi_rag_lab.eval.judge.items import read_items
from fastapi_rag_lab.eval.judge.judge import LABELS, SUPPORTED, UNSUPPORTED
from fastapi_rag_lab.eval.judge.labels import read_labels

if TYPE_CHECKING:
    from collections.abc import Mapping


def compared_items(
    human: Mapping[str, str], runs: Mapping[str, Mapping[str, str]]
) -> list[str]:
    """Items the human labelled unsupported that every run gave a verdict for."""
    return sorted(
        i
        for i, label in human.items()
        if label == UNSUPPORTED and all(run.get(i) in LABELS for run in runs.values())
    )


def gate_report(
    human: Mapping[str, str],
    baseline: Mapping[str, Mapping[str, str]],
    candidate: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if not baseline or not candidate:
        raise ValueError("the gate needs at least one baseline and one candidate run")
    if baseline.keys() & candidate.keys():
        raise ValueError("a run cannot be both baseline and candidate")
    runs = {**baseline, **candidate}
    ids = compared_items(human, runs)
    if not ids:
        raise ValueError("no item is labelled unsupported and judged by every run")

    def rates(group: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, Any]]:
        out = {}
        for name, labels in group.items():
            passed = [i for i in ids if labels[i] == SUPPORTED]
            out[name] = {"false_pass": len(passed), "rate": len(passed) / len(ids)}
        return out

    base, cand = rates(baseline), rates(candidate)
    ceiling = max(r["rate"] for r in base.values())
    mean = sum(r["rate"] for r in cand.values()) / len(cand)
    return {
        "n_items": len(ids),
        "items": ids,
        "baseline": base,
        "candidate": cand,
        "baseline_range": [min(r["rate"] for r in base.values()), ceiling],
        "candidate_mean": mean,
        "passed": mean <= ceiling,
        "rule": (
            "candidate mean false-pass rate <= highest baseline run's rate, on the "
            "items the human labelled unsupported and every run judged"
        ),
    }


def read_run(path: Path) -> tuple[dict[str, str], str | None]:
    """A judge run's verdicts and its judge_config_sha256 (None in older runs)."""
    labels = {j["item_id"]: j["label"] for j in read_items(path / "judgements.jsonl")}
    summary_path = path / "summary.json"
    config = None
    if summary_path.exists():
        config = json.loads(summary_path.read_text()).get("judge_config_sha256")
    return labels, config


def format_report(report: Mapping[str, Any]) -> str:
    lines = [f"items labelled unsupported and judged by every run: {report['n_items']}"]
    for group in ("baseline", "candidate"):
        for name, r in report[group].items():
            lines.append(
                f"{group:<9} {name:<50} false pass {r['false_pass']:>2}"
                f" ({r['rate']:.0%})"
            )
    low, high = report["baseline_range"]
    lines.append(
        f"baseline range {low:.0%} to {high:.0%}; "
        f"candidate mean {report['candidate_mean']:.0%}: "
        + ("PASS" if report["passed"] else "FAIL")
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--labels", type=Path, help="default: labels.csv in the items run")
    p.add_argument("--baseline", type=Path, action="append", required=True)
    p.add_argument("--candidate", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    ids = {i["item_id"] for i in read_items(args.items / "items.jsonl")}
    try:
        human = read_labels(args.labels or args.items / "labels.csv", ids)
        baseline = {path.name: read_run(path) for path in args.baseline}
        candidate = {path.name: read_run(path) for path in args.candidate}
        configs = {
            group: sorted({c for _, c in runs.values() if c is not None})
            for group, runs in (("baseline", baseline), ("candidate", candidate))
        }
        for group, found in configs.items():
            if len(found) > 1:
                raise ValueError(f"{group} runs have different judge configs: {found}")
        if configs["baseline"] and configs["baseline"] == configs["candidate"]:
            raise ValueError("baseline and candidate have the same judge config")
        report = gate_report(
            human,
            {name: labels for name, (labels, _) in baseline.items()},
            {name: labels for name, (labels, _) in candidate.items()},
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report["items_run"] = args.items.name
    report["judge_config_sha256"] = {
        group: found[0] if found else None for group, found in configs.items()
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
