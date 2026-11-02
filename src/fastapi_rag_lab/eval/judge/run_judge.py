"""Run the faithfulness judge over an items run (see ``answers.py``). Spends money.

One Claude call per item. ``--budget-usd`` is a hard cap checked before every
call (overshoot at most one call); reaching it stops the run and keeps what was
judged. Writes, into a new ``<out-root>/<run-id>/``:

- ``judgements.jsonl``  item id, label, rationale, tokens, cost, latency
                        (+ ``raw_reply``, whole, when the reply was unparsed)
- ``llm_calls.jsonl``   one line per API call, with the reply text
- ``summary.json``      label counts, totals, model, prompt, budget, git,
                        items run and its sha256, pre-registered labels hash

    uv run --extra judge python -m fastapi_rag_lab.eval.judge.run_judge \\
        --items reports/2026-10-30-judge-items-so-dev-50 --model haiku \\
        --budget-usd 0.10 --run-id 2026-10-30-judge-haiku-so-dev-50

``--prompt v1`` (default) is the frozen calibration prompt; ``v2`` adds the
citation rule. ``--effort`` overrides the model's default (``low``).
``--structured`` forces a schema-checked tool call instead of free JSON.
``--preregistered-labels`` records the sha256 of a labels file committed
BEFORE this run, so a reader can check the labels did not change after the
judge was seen.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi_rag_lab.eval.judge.items import read_items
from fastapi_rag_lab.eval.judge.judge import (
    PROMPTS,
    UNPARSED,
    VERDICT_TOOL,
    judge_user,
    parse_verdict,
)
from fastapi_rag_lab.eval.judge.llm import (
    FAILURE_STOP_REASONS,
    MODELS,
    BudgetedClient,
    BudgetExceeded,
)
from fastapi_rag_lab.eval.provenance import git_info

ERROR = "error"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def judge_items(
    items: list[dict[str, Any]],
    client: BudgetedClient,
    *,
    system: str,
    structured: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """Judge items in order; stop at the budget. Returns judgements and stop reason."""
    out: list[dict[str, Any]] = []
    tool = VERDICT_TOOL if structured else None
    for item in items:
        user = judge_user(
            item["question"],
            [s["parent_text"] for s in item["sources"]],
            item["answer"],
        )
        try:
            text, usage = client.complete(
                system, user, item_id=item["item_id"], tool=tool
            )
        except BudgetExceeded as exc:
            return out, str(exc)
        except Exception as exc:  # API failure: recorded, not guessed
            out.append(
                {
                    "item_id": item["item_id"],
                    "label": ERROR,
                    "rationale": repr(exc)[:300],
                }
            )
            continue
        if usage.stop_reason in FAILURE_STOP_REASONS:
            verdict_label, rationale = UNPARSED, f"stop reason {usage.stop_reason}"
        else:
            verdict = parse_verdict(text)
            verdict_label, rationale = verdict.label, verdict.rationale
        row: dict[str, Any] = {
            "item_id": item["item_id"],
            "label": verdict_label,
            "rationale": rationale,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": usage.cost_usd,
            "latency_ms": round(usage.latency_ms, 1),
        }
        if verdict_label == UNPARSED:
            row["raw_reply"] = text  # whole, so the failure can be read back
        out.append(row)
    return out, None


def main(argv: list[str] | None = None, *, sdk_client: Any = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--items", type=Path, required=True, help="items run directory")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-root", type=Path, default=Path("reports"))
    p.add_argument("--prompt", choices=sorted(PROMPTS), default="v1")
    p.add_argument("--effort", choices=["low", "medium", "high"], default=None)
    p.add_argument("--structured", action="store_true", help="forced tool call")
    p.add_argument(
        "--preregistered-labels",
        type=Path,
        default=None,
        help="labels.csv committed before this run; its sha256 goes in the summary",
    )
    args = p.parse_args(argv)

    items = read_items(args.items / "items.jsonl")
    if not items:
        print(f"error: no items in {args.items}", file=sys.stderr)
        return 2
    run_dir = args.out_root / args.run_id
    if run_dir.exists():
        print(f"error: {run_dir} exists; a judge run id must be new", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)

    model = MODELS[args.model]
    if args.effort:
        model = dataclasses.replace(model, effort=args.effort)
    system = PROMPTS[args.prompt]
    client = BudgetedClient(
        model,
        args.budget_usd,
        log_path=run_dir / "llm_calls.jsonl",
        sdk_client=sdk_client,
    )
    started_at = datetime.now(UTC).isoformat()
    judgements, stopped = judge_items(
        items, client, system=system, structured=args.structured
    )
    with (run_dir / "judgements.jsonl").open("w", encoding="utf-8") as fh:
        for j in judgements:
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")
    summary = {
        "run_id": args.run_id,
        "items_run": args.items.name,
        "items_sha256": sha256_of(args.items / "items.jsonl"),
        "preregistered_labels": (
            {
                "path": str(args.preregistered_labels),
                "sha256": sha256_of(args.preregistered_labels),
            }
            if args.preregistered_labels
            else None
        ),
        "n_items": len(items),
        "n_judged": len(judgements),
        "labels": dict(Counter(j["label"] for j in judgements)),
        "stopped": stopped,
        "cost_usd": client.spent,
        "budget_usd": args.budget_usd,
        "input_tokens": sum(j.get("input_tokens", 0) for j in judgements),
        "output_tokens": sum(j.get("output_tokens", 0) for j in judgements),
        "model": {
            "id": model.model,
            "effort": model.effort,
            "max_tokens": model.max_tokens,
            "input_per_mtok": model.input_per_mtok,
            "output_per_mtok": model.output_per_mtok,
        },
        "judge_prompt_version": args.prompt,
        "judge_system_prompt": system,
        "structured_output": args.structured,
        "command": " ".join(
            ["run_judge.py", *(sys.argv[1:] if argv is None else argv)]
        ),
        "git": git_info(),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{summary['labels']} cost ${client.spent:.4f} (budget ${args.budget_usd})")
    if stopped:
        print(f"stopped early: {stopped}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
