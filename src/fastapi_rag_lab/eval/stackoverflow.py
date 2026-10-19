"""Build a retrieval eval set from real Stack Overflow questions.

Questions tagged ``fastapi`` are the queries (title only). Labels are the
FastAPI docs pages linked from answers that were accepted or scored >= 1,
resolved against the pinned docs corpus. Output matches the golden dataset
schema (``category="stackoverflow"``) plus a provenance file.

    python -m fastapi_rag_lab.eval.stackoverflow build \
        --docs-dir data/raw/fastapi_docs --out tests/eval/stackoverflow
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx

API = "https://api.stackexchange.com/2.3"
SITE = "stackoverflow"
TAG = "fastapi"
CACHE_DIR = Path("data/raw/stackoverflow")
MANIFEST_PATH = Path("data/raw/manifest.json")
LICENSE_NAME = "CC BY-SA 4.0"
# Stack Overflow switched user contributions to CC BY-SA 4.0 on 2018-05-02.
LICENSE_4_0_FROM = datetime(2018, 5, 2, tzinfo=UTC)
MIN_QUOTA = 30
SLEEP_S = 0.3
MAX_RETRIES = 2
CATEGORY = "stackoverflow"

# Language directories used by the FastAPI docs translations (docs/<lang>/).
# Explicit rather than "any 2-letter segment" so a real English section can
# never be mistaken for a language. Stripping is skipped if the first segment
# is itself a top-level entry of the corpus.
_LANGS = (
    "en de es fr it pt ru uk tr ja ko zh zh-hant pl vi id he ar fa hu bn nl sv el "
    "cs da fi ro th bg hr sk sl sr ca lt lv et hi ur ms"
)
LANG_PREFIXES = frozenset(_LANGS.split())

_LINK_RE = re.compile(
    r"https?://(?:www\.)?fastapi\.tiangolo\.com(/[^\s\"'<>)\]]*)?", re.IGNORECASE
)
_DOC_EXTS = {".md", ".html", ""}


# ---------------------------------------------------------------- labels


def extract_doc_paths(text: str) -> list[str]:
    """Return normalized URL paths of every docs link."""
    out = []
    for m in _LINK_RE.finditer(text):
        raw = (m.group(1) or "/").rstrip(".,;:!?")
        path = urlsplit("https://x" + raw).path
        out.append(path.strip("/"))
    return out


def strip_lang_prefix(path: str, docs_dir: Path | None = None) -> str:
    parts = path.split("/") if path else []
    if parts and parts[0].lower() in LANG_PREFIXES:
        if docs_dir is not None and (docs_dir / parts[0]).exists():
            return path
        return "/".join(parts[1:])
    return path


def resolve_path(path: str, docs_dir: Path) -> str | None:
    """Map a URL path to a corpus-relative markdown file, or None."""
    path = strip_lang_prefix(path, docs_dir)
    if path == "":
        cands = ["index.md"]
    else:
        if ".." in path.split("/"):
            return None
        cands = [f"{path}.md", f"{path}/index.md"]
    for c in cands:
        if (docs_dir / c).is_file():
            return c
    return None


def is_doc_path(path: str) -> bool:
    return Path(path).suffix.lower() in _DOC_EXTS


def qualifying(answer: dict[str, Any]) -> bool:
    return bool(answer.get("is_accepted")) or answer.get("score", 0) >= 1


def label_question(
    answers: list[dict[str, Any]], docs_dir: Path, stats: Counter, unresolved: Counter
) -> tuple[list[str], list[int]]:
    """Return (sorted labels, answer ids that provided labels) for one question."""
    labels: set[str] = set()
    ans_ids: list[int] = []
    for a in answers:
        if not qualifying(a):
            continue
        provided = False
        paths = extract_doc_paths(html.unescape(a.get("body", "")))
        if paths:
            stats["qualifying_answers_with_link"] += 1
        for p in paths:
            stats["links"] += 1
            norm = strip_lang_prefix(p, docs_dir)
            if norm == "release-notes" or not is_doc_path(norm):
                stats["links_excluded"] += 1
                continue
            res = resolve_path(p, docs_dir)
            if res is None or res == "release-notes.md":
                stats["links_unresolved"] += 1
                unresolved[norm or "/"] += 1
                continue
            stats["links_resolved"] += 1
            labels.add(res)
            provided = True
        if provided:
            ans_ids.append(a["answer_id"])
    return sorted(labels), sorted(ans_ids)


# ----------------------------------------------------------------- split


def split_of(question_id: int, seed: int, dev_fraction: float) -> str:
    h = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    return "dev" if u < dev_fraction else "test"


# ----------------------------------------------------------------- fetch


class QuotaLowError(RuntimeError):
    pass


class Fetcher:
    """Cached Stack Exchange API client. Cache files hold the decoded JSON."""

    def __init__(self, cache_dir: Path, refresh: bool = False) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.requests = 0
        self.quota_remaining: int | None = None
        self.fetched_at: list[str] = []
        self._client: httpx.Client | None = None

    def _key(self, endpoint: str, params: dict[str, Any]) -> Path:
        q = urlencode(sorted(params.items()))
        digest = hashlib.sha256(f"{endpoint}?{q}".encode()).hexdigest()[:16]
        name = "answers" if endpoint.endswith("/answers") else "questions"
        return self.cache_dir / f"{name}_{digest}.json"

    def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "site": SITE}
        path = self._key(endpoint, params)
        if path.exists() and not self.refresh:
            cached = json.loads(path.read_text(encoding="utf-8"))
            self.fetched_at.append(cached["fetched_at"])
            return cached["response"]
        if self.quota_remaining is not None and self.quota_remaining < MIN_QUOTA:
            raise QuotaLowError(f"quota_remaining={self.quota_remaining} < {MIN_QUOTA}")
        if self._client is None:
            self._client = httpx.Client(
                timeout=30,
                headers={"Accept-Encoding": "gzip", "User-Agent": "fastapi-rag-lab"},
            )
        for attempt in range(MAX_RETRIES + 1):
            self.requests += 1
            resp = self._client.get(API + endpoint, params=params)
            if resp.status_code == 200:
                break
            if attempt == MAX_RETRIES or resp.status_code < 500:
                raise RuntimeError(
                    f"{endpoint}: HTTP {resp.status_code} {resp.text[:200]}"
                )
            time.sleep(2 * (attempt + 1))
        data = resp.json()
        self.quota_remaining = data.get("quota_remaining")
        now = datetime.now(UTC).isoformat(timespec="seconds")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(
            {
                "fetched_at": now,
                "endpoint": endpoint,
                "params": params,
                "response": data,
            }
        )
        path.write_bytes(blob.encode("utf-8"))
        self.fetched_at.append(now)
        time.sleep(max(SLEEP_S, float(data.get("backoff", 0))))
        if self.quota_remaining is not None and self.quota_remaining < MIN_QUOTA:
            raise QuotaLowError(f"quota_remaining={self.quota_remaining} < {MIN_QUOTA}")
        return data


def fetch_questions(f: Fetcher, max_questions: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    page = 1
    while len(out) < max_questions:
        data = f.get(
            "/questions",
            {
                "tagged": TAG,
                "sort": "votes",
                "order": "desc",
                "pagesize": 100,
                "page": page,
            },
        )
        out.extend(data["items"])
        if not data.get("has_more"):
            break
        page += 1
    return out[:max_questions]


def fetch_answers(f: Fetcher, ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    by_q: dict[int, list[dict[str, Any]]] = {i: [] for i in ids}
    for i in range(0, len(ids), 100):
        chunk = ";".join(str(x) for x in ids[i : i + 100])
        page = 1
        while True:
            data = f.get(
                f"/questions/{chunk}/answers",
                {
                    "order": "desc",
                    "sort": "votes",
                    "pagesize": 100,
                    "page": page,
                    "filter": "withbody",
                },
            )
            for a in data["items"]:
                by_q.setdefault(a["question_id"], []).append(a)
            if not data.get("has_more"):
                break
            page += 1
    return by_q


# ----------------------------------------------------------------- build


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus_commit() -> str:
    try:
        m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for k in ("commit_sha", "corpus_commit", "commit", "sha"):
            if k in m:
                return str(m[k])
        for v in m.values():
            if isinstance(v, dict):
                for k in ("commit_sha", "corpus_commit", "commit", "sha"):
                    if k in v:
                        return str(v[k])
    except (OSError, ValueError):
        pass
    from fastapi_rag_lab.ingest.fetcher import PINNED_COMMIT_SHA

    return PINNED_COMMIT_SHA


def build_items(
    questions: list[dict[str, Any]],
    answers: dict[int, list[dict[str, Any]]],
    docs_dir: Path,
    seed: int,
    dev_fraction: float,
) -> tuple[list[dict[str, Any]], Counter, Counter]:
    stats: Counter = Counter()
    unresolved: Counter = Counter()
    items = []
    for q in questions:
        qid = q["question_id"]
        created = datetime.fromtimestamp(q["creation_date"], UTC)
        assert created > LICENSE_4_0_FROM, f"question {qid} predates CC BY-SA 4.0"
        ans = answers.get(qid, [])
        stats["answers"] += len(ans)
        stats["qualifying_answers"] += sum(qualifying(a) for a in ans)
        labels, ans_ids = label_question(ans, docs_dir, stats, unresolved)
        if not labels:
            continue
        owner = q.get("owner", {})
        items.append(
            {
                "id": f"so_{qid}",
                "split": split_of(qid, seed, dev_fraction),
                "query": html.unescape(q["title"]),
                "expected_sources": labels,
                "question_id": qid,
                "link": q["link"],
                "score": q["score"],
                "creation_date": created.isoformat(),
                "author_display_name": html.unescape(owner.get("display_name", ""))
                or None,
                "author_link": owner.get("link"),
                "label_answer_ids": ans_ids,
                "accepted_answer_id": q.get("accepted_answer_id"),
                "license": LICENSE_NAME,
            }
        )
    items.sort(key=lambda x: x["question_id"])
    return items, stats, unresolved


LICENSE_MD = """# Stack Overflow derived data

The files in this directory (`dev.json`, `test.json`, `questions.jsonl`) contain
question titles and metadata from Stack Overflow (https://stackoverflow.com),
obtained through the Stack Exchange API. User contributions on Stack Overflow
are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
This derived data is distributed under the same license.

- Source: questions tagged `fastapi` on Stack Overflow, created after
  2018-05-02 (when CC BY-SA 4.0 took effect).
- Attribution: every item in `questions.jsonl` carries the question `link` and
  the author (`author_display_name`, `author_link`). Retain these when reusing.
- Changes made: only question titles are kept (no question or answer bodies);
  titles are HTML-unescaped. Labels (FastAPI docs pages) were derived from
  links found in answers, which are not redistributed.
"""


def build(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        print(f"docs dir not found: {docs_dir}", file=sys.stderr)
        return 2
    out = Path(args.out)
    f = Fetcher(Path(args.cache_dir), refresh=args.refresh)
    try:
        questions = fetch_questions(f, args.max_questions)
        answers = fetch_answers(f, [q["question_id"] for q in questions])
    except QuotaLowError as e:
        print(f"aborting cleanly, no outputs written: {e}", file=sys.stderr)
        return 3

    items, stats, unresolved = build_items(
        questions, answers, docs_dir, args.seed, args.dev_fraction
    )
    out.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        entries = [
            {
                "id": i["id"],
                "query": i["query"],
                "expected_sources": i["expected_sources"],
                "category": CATEGORY,
            }
            for i in items
            if i["split"] == split
        ]
        (out / f"{split}.json").write_text(
            json.dumps({"entries": entries}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (out / "questions.jsonl").write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
        encoding="utf-8",
    )
    (out / "LICENSE.md").write_text(LICENSE_MD, encoding="utf-8")

    qs_with_link = sum(
        1
        for q in questions
        if any(
            extract_doc_paths(html.unescape(a.get("body", "")))
            for a in answers.get(q["question_id"], [])
            if qualifying(a)
        )
    )
    hist = Counter(len(i["expected_sources"]) for i in items)
    report = {
        "snapshot_utc": max(f.fetched_at) if f.fetched_at else None,
        "api": {
            "base": API,
            "site": SITE,
            "tagged": TAG,
            "sort": "votes",
            "order": "desc",
            "question_pagesize": 100,
            "answers_filter": "withbody",
        },
        "requests_made_this_run": f.requests,
        "cached_responses": len(list(Path(args.cache_dir).glob("*.json"))),
        "quota_remaining_last_seen": f.quota_remaining,
        "params": {
            "max_questions": args.max_questions,
            "dev_fraction": args.dev_fraction,
            "seed": args.seed,
        },
        "counts": {
            "questions_fetched": len(questions),
            "answers_fetched": stats["answers"],
            "qualifying_answers": stats["qualifying_answers"],
            "qualifying_answers_with_docs_link": stats["qualifying_answers_with_link"],
            "questions_with_docs_link_in_qualifying_answer": qs_with_link,
            "docs_links_total": stats["links"],
            "docs_links_excluded_release_notes_or_assets": stats["links_excluded"],
            "docs_links_resolved": stats["links_resolved"],
            "docs_links_unresolved": stats["links_unresolved"],
            "questions_kept": len(items),
            "dev": sum(i["split"] == "dev" for i in items),
            "test": sum(i["split"] == "test" for i in items),
        },
        "labels_per_question_histogram": {str(k): hist[k] for k in sorted(hist)},
        "unresolved_links": {
            "total": stats["links_unresolved"],
            "distinct_paths": len(unresolved),
            "top20": [{"path": p, "count": c} for p, c in unresolved.most_common(20)],
        },
        "corpus_commit_sha": _corpus_commit(),
        "sha256": {
            n: _sha(out / n) for n in ("dev.json", "test.json", "questions.jsonl")
        },
        "command": "python -m fastapi_rag_lab.eval.stackoverflow "
        + " ".join(sys.argv[1:]),
    }
    (out / "build.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--docs-dir", default="data/raw/fastapi_docs")
    b.add_argument("--out", default="tests/eval/stackoverflow")
    b.add_argument("--cache-dir", default=str(CACHE_DIR))
    b.add_argument("--max-questions", type=int, default=1500)
    b.add_argument("--dev-fraction", type=float, default=0.4)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
