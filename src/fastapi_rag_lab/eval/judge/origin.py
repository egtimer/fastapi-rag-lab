"""Where do unsupported claims come from: the corpus, the model, or nowhere?

The judge says an answer is ``unsupported`` by its five sources. That is one
of several different failures:

- ``in_sources_contradicted``   the claim contradicts a retrieved source
                                (generation failure, confident)
- ``in_corpus_not_retrieved``   the claim IS in the FastAPI docs, in a chunk
                                that was not among the five sources
                                (retrieval failure; the model may also know
                                it from its weights)
- ``true_not_in_corpus``        true of FastAPI but not in the docs corpus
                                (from the weights)
- ``false``                     not true of FastAPI (fabrication)
- ``about_sources``             a false claim about the sources themselves
                                (invented citation, misattribution)
- ``unclear``

This module does the mechanical part: for every claim the human labeller
noted on an ``unsupported`` item, it searches the WHOLE chunk corpus with
BM25 (the same tokenizer as sparse retrieval) and writes a review sheet with
the best-matching sections, marking which were among the item's retrieved
sources. The human then writes one origin per claim in ``origin_labels.csv``.
``--summarize`` counts the labelled origins.

    uv run python -m fastapi_rag_lab.eval.judge.origin \\
        --items reports/2026-10-30-judge-items-so-dev-50 \\
        --out reports/<date>-unsupported-origin-so-dev-50
    # ... fill origin_labels.csv ...
    uv run python -m fastapi_rag_lab.eval.judge.origin \\
        --summarize reports/<date>-unsupported-origin-so-dev-50

Needs the docs corpus on disk (``data/raw/fastapi_docs``, from the ingest
fetcher at the pinned commit); no Ollama, no Qdrant.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi_rag_lab.eval.judge.items import read_items
from fastapi_rag_lab.eval.judge.judge import UNSUPPORTED
from fastapi_rag_lab.eval.judge.labels import read_labels

ORIGINS = (
    "in_sources_contradicted",
    "in_corpus_not_retrieved",
    "true_not_in_corpus",
    "false",
    "about_sources",
    "unclear",
)
TOP_SECTIONS = 3
SNIPPET_CHARS = 500


@dataclass(frozen=True)
class Section:
    source_file: str
    heading_path: tuple[str, ...]
    text: str

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return (self.source_file, self.heading_path)


def _tokenize(text: str) -> list[str]:
    """Same as ``retrieval.sparse._tokenize``: lowercase, ``\\w+`` tokens."""
    return re.findall(r"\w+", text.lower())


def load_corpus(docs_dir: Path) -> tuple[list[Section], list[list[str]]]:
    """Parent sections of every markdown file, chunked as ingestion does, and
    the BM25 documents (one per CHILD chunk, mapped back to its parent)."""
    from fastapi_rag_lab.ingest.chunker import chunk_markdown

    sections: list[Section] = []
    docs: list[list[str]] = []
    owner: list[int] = []
    root = docs_dir.resolve()
    for path in sorted(root.rglob("*.md")):
        rel = str(path.relative_to(root))
        parents, children = chunk_markdown(rel, path.read_text(encoding="utf-8"))
        index = {}
        for parent in parents:
            index[parent.parent_id] = len(sections)
            sections.append(
                Section(parent.source_file, tuple(parent.heading_path), parent.text)
            )
        for child in children:
            docs.append(_tokenize(child.text))
            owner.append(index[child.parent_id])
    return sections, [[str(o)] + d for o, d in zip(owner, docs, strict=True)]


def split_claims(note: str) -> list[str]:
    """The labeller's note, one claim per ``;``; empty notes give no claims."""
    return [c.strip() for c in note.split(";") if c.strip()]


def search(
    bm25: Any, owners: list[int], sections: list[Section], query: str, k: int
) -> list[tuple[Section, float]]:
    """Top ``k`` sections for ``query``, best child score per section."""
    scores = bm25.get_scores(_tokenize(query))
    best: dict[int, float] = {}
    for i, s in enumerate(scores):
        if s <= 0:
            continue
        o = owners[i]
        if s > best.get(o, 0.0):
            best[o] = float(s)
    ranked = sorted(best.items(), key=lambda kv: -kv[1])[:k]
    return [(sections[o], s) for o, s in ranked]


def build(items_dir: Path, docs_dir: Path, out_dir: Path) -> int:
    from rank_bm25 import BM25Okapi

    items = read_items(items_dir / "items.jsonl")
    ids = {i["item_id"] for i in items}
    labels = read_labels(items_dir / "labels.csv", ids)
    notes: dict[str, str] = {}
    with (items_dir / "labels.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            notes[row["item_id"].strip()] = (row.get("note") or "").strip()

    sections, tagged = load_corpus(docs_dir)
    owners = [int(d[0]) for d in tagged]
    bm25 = BM25Okapi([d[1:] for d in tagged])

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = [
        "# Where do the unsupported claims come from?",
        "",
        "For each claim, read the candidate sections (whole corpus, BM25) and",
        "write ONE origin per claim in `origin_labels.csv`:",
        "",
        *[f"- `{o}`" for o in ORIGINS],
        "",
        "`retrieved: yes` means the section was among the five sources the",
        "generator saw. A claim that matches a `retrieved: no` section is a",
        "retrieval failure, not (only) a generation failure.",
        "",
    ]
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    n_items = 0
    for item in items:
        iid = item["item_id"]
        if labels.get(iid) != UNSUPPORTED:
            continue
        n_items += 1
        retrieved = {
            (s.get("source_file", ""), tuple(s.get("heading_path") or []))
            for s in item["sources"]
        }
        sheet += [f"## {iid}: {item['question'].strip()}", ""]
        sheet += ["**Answer (first 600 chars):**", "", item["answer"].strip()[:600], ""]
        claims = split_claims(notes.get(iid, "")) or ["(no note: read the answer)"]
        for n, claim in enumerate(claims, 1):
            hits = search(bm25, owners, sections, claim, TOP_SECTIONS)
            sheet += [f"### claim {n}: {claim}", ""]
            cand = {"item_id": iid, "claim_no": n, "claim": claim, "sections": []}
            for sec, score in hits:
                was = "yes" if sec.key in retrieved else "no"
                where = " > ".join(sec.heading_path) or sec.source_file
                sheet += [
                    f"- **{sec.source_file}** ({where}), bm25 {score:.1f}, "
                    f"retrieved: {was}",
                    "",
                    "  > " + sec.text.strip()[:SNIPPET_CHARS].replace("\n", "\n  > "),
                    "",
                ]
                cand["sections"].append(
                    {
                        "source_file": sec.source_file,
                        "heading_path": list(sec.heading_path),
                        "bm25": round(score, 2),
                        "retrieved": was == "yes",
                    }
                )
            candidates.append(cand)
            rows.append(
                {
                    "item_id": iid,
                    "claim_no": n,
                    "claim": claim,
                    "origin": "",
                    "evidence": "",
                }
            )
        sheet.append("---\n")
    (out_dir / "origin_sheet.md").write_text("\n".join(sheet), encoding="utf-8")
    with (out_dir / "candidates.jsonl").open("w", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    csv_path = out_dir / "origin_labels.csv"
    if csv_path.exists():
        print(f"{csv_path} exists; not overwriting labels", file=sys.stderr)
    else:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(
        f"{n_items} unsupported items, {len(rows)} claims, "
        f"{len(sections)} corpus sections; sheet in {out_dir}"
    )
    return 0


def summarize(out_dir: Path) -> int:
    with (out_dir / "origin_labels.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    labelled = [r for r in rows if r["origin"].strip()]
    bad = [r for r in labelled if r["origin"].strip() not in ORIGINS]
    if bad:
        print(f"error: unknown origin in {[(r['item_id'], r['origin']) for r in bad]}")
        return 2
    by_claim = Counter(r["origin"].strip() for r in labelled)
    per_item: dict[str, set[str]] = {}
    for r in labelled:
        per_item.setdefault(r["item_id"], set()).add(r["origin"].strip())
    by_item = Counter(o for origins in per_item.values() for o in origins)
    summary = {
        "claims_labelled": len(labelled),
        "claims_total": len(rows),
        "items": len(per_item),
        "by_claim": dict(by_claim),
        "items_with_origin": dict(by_item),
        "items_only_retrieval_failure": sum(
            1 for s in per_item.values() if s == {"in_corpus_not_retrieved"}
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--items", type=Path, help="items run with labels.csv")
    p.add_argument("--docs", type=Path, default=Path("data/raw/fastapi_docs"))
    p.add_argument("--out", type=Path, help="output directory")
    p.add_argument("--summarize", type=Path, help="count origins in this out dir")
    args = p.parse_args(argv)
    if args.summarize:
        return summarize(args.summarize)
    if not args.items or not args.out:
        p.error("--items and --out are required unless --summarize is given")
    if not args.docs.exists():
        print(f"error: {args.docs} missing; run the ingest fetcher", file=sys.stderr)
        return 2
    return build(args.items, args.docs, args.out)


if __name__ == "__main__":
    sys.exit(main())
