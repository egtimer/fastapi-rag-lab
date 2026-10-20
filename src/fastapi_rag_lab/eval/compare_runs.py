"""Compare the retrieved source lists of two benchmark runs, query by query.

For every config the two runs share, counts the queries whose ordered
``retrieved_sources`` are identical and prints the mean Recall@k of each run.
Used to check that a run reproduces another one (same code, corpus and
runtime should give the same lists).

    uv run python -m fastapi_rag_lab.eval.compare_runs reports/<a> reports/<b>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConfigComparison:
    config: str
    n: int
    identical: int
    recall_a: float
    recall_b: float


def _rows(run_dir: Path) -> dict[tuple[str, str], dict]:
    report = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
    return {(r["config_label"], r["query_id"]): r for r in report["per_query"]}


def compare(run_a: Path, run_b: Path) -> list[ConfigComparison]:
    a, b = _rows(run_a), _rows(run_b)
    configs = sorted({c for c, _ in a} & {c for c, _ in b})
    out = []
    for config in configs:
        queries = sorted(q for c, q in a.keys() & b.keys() if c == config)
        n = len(queries)
        out.append(
            ConfigComparison(
                config=config,
                n=n,
                identical=sum(
                    a[config, q]["retrieved_sources"]
                    == b[config, q]["retrieved_sources"]
                    for q in queries
                ),
                recall_a=sum(a[config, q]["recall_at_k"] for q in queries) / n,
                recall_b=sum(b[config, q]["recall_at_k"] for q in queries) / n,
            )
        )
    return out


def format_markdown(rows: list[ConfigComparison], run_a: Path, run_b: Path) -> str:
    lines = [
        f"a: `{run_a}`  b: `{run_b}`",
        "",
        "| Config | Identical source lists | Recall@k a | Recall@k b |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r.config} | {r.identical}/{r.n} | {r.recall_a:.3f} | {r.recall_b:.3f} |"
        for r in rows
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    args = p.parse_args(argv)
    rows = compare(args.run_a, args.run_b)
    if not rows:
        print("error: the runs share no config", file=sys.stderr)
        return 2
    print(format_markdown(rows, args.run_a, args.run_b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
