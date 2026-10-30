"""Run the faithfulness judge over an items run (see ``answers.py``). Spends money.

One Claude call per item. ``--budget-usd`` is a hard cap checked before every
call (overshoot at most one call); reaching it stops the run and keeps what was
judged. Writes, into a new ``<out-root>/<run-id>/``:

- ``judgements.jsonl``  item id, label, rationale, tokens, cost, latency
- ``llm_calls.jsonl``   one line per API call
- ``summary.json``      label counts, totals, model, budget, git, items run

    uv run --extra judge python -m fastapi_rag_lab.eval.judge.run_judge \\
        --items reports/2026-10-30-judge-items-so-dev-50 --model haiku \\
        --budget-usd 0.10 --run-id 2026-10-30-judge-haiku-so-dev-50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi_rag_lab.eval.judge.answers import read_items
from fastapi_rag_lab.eval.judge.judge import (
    JUDGE_SYSTEM,
    UNPARSED,
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


def judge_items(
    items: list[dict[str, Any]], client: BudgetedClient
) -> tuple[list[dict[str, Any]], str | None]:
    """Judge items in order; stop at the budget. Returns judgements and stop reason."""
    out: list[dict[str, Any]] = []
    for item in items:
        user = judge_user(
            item["question"],
            [s["parent_text"] for s in item["sources"]],
            item["answer"],
        )
        try:
            text, usage = client.complete(JUDGE_SYSTEM, user, item_id=item["item_id"])
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
        out.append(
            {
                "item_id": item["item_id"],
                "label": verdict_label,
                "rationale": rationale,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": usage.cost_usd,
                "latency_ms": round(usage.latency_ms, 1),
            }
        )
    return out, None


def main(argv: list[str] | None = None, *, sdk_client: Any = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--items", type=Path, required=True, help="items run directory")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-root", type=Path, default=Path("reports"))
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
    client = BudgetedClient(
        model,
        args.budget_usd,
        log_path=run_dir / "llm_calls.jsonl",
        sdk_client=sdk_client,
    )
    started_at = datetime.now(UTC).isoformat()
    judgements, stopped = judge_items(items, client)
    with (run_dir / "judgements.jsonl").open("w", encoding="utf-8") as fh:
        for j in judgements:
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")
    summary = {
        "run_id": args.run_id,
        "items_run": args.items.name,
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
        "judge_system_prompt": JUDGE_SYSTEM,
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
