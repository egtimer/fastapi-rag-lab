"""Offline tests for the Stack Overflow eval-set builder (no network)."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

import pytest

from fastapi_rag_lab.eval.dataset import load_golden_dataset
from fastapi_rag_lab.eval.stackoverflow import (
    build_items,
    extract_doc_paths,
    label_question,
    qualifying,
    resolve_path,
    split_of,
    strip_lang_prefix,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    (tmp_path / "x.md").write_text("x")
    (tmp_path / "y").mkdir()
    (tmp_path / "y" / "index.md").write_text("y")
    (tmp_path / "index.md").write_text("home")
    (tmp_path / "release-notes.md").write_text("rn")
    return tmp_path


def test_extract_links_drops_fragment_query_and_punctuation():
    text = (
        'see <a href="https://fastapi.tiangolo.com/x/#frag">x</a>, '
        "and http://fastapi.tiangolo.com/y/?q=1. Also https://fastapi.tiangolo.com "
        "but not https://example.com/x/"
    )
    assert extract_doc_paths(text) == ["x", "y", ""]


def test_strip_language_prefix(docs: Path):
    assert strip_lang_prefix("es/x") == "x"
    assert strip_lang_prefix("zh-hant/y") == "y"
    assert strip_lang_prefix("es") == ""
    assert strip_lang_prefix("tutorial/x") == "tutorial/x"
    # a real corpus directory is never treated as a language
    (docs / "de").mkdir()
    assert strip_lang_prefix("de/x", docs) == "de/x"


def test_resolve_path(docs: Path):
    assert resolve_path("x", docs) == "x.md"
    assert resolve_path("y", docs) == "y/index.md"
    assert resolve_path("es/x", docs) == "x.md"
    assert resolve_path("", docs) == "index.md"
    assert resolve_path("missing", docs) is None
    assert resolve_path("../x", docs) is None


def _ans(aid, body, score=0, accepted=False):
    return {"answer_id": aid, "body": body, "score": score, "is_accepted": accepted}


def test_qualifying_filter():
    assert qualifying(_ans(1, "", score=1))
    assert qualifying(_ans(1, "", score=0, accepted=True))
    assert not qualifying(_ans(1, "", score=0))
    assert not qualifying(_ans(1, "", score=-3))


def test_label_question_filters_and_reports_unresolved(docs: Path):
    answers = [
        _ans(1, "https://fastapi.tiangolo.com/x/", score=2),
        _ans(2, "https://fastapi.tiangolo.com/y/", score=0),  # not qualifying
        _ans(
            3,
            "https://fastapi.tiangolo.com/nope/ https://fastapi.tiangolo.com/es/y/",
            accepted=True,
        ),
        _ans(
            4,
            "https://fastapi.tiangolo.com/release-notes/ "
            "https://fastapi.tiangolo.com/img/logo.png",
            score=5,
        ),
    ]
    stats: Counter = Counter()
    unresolved: Counter = Counter()
    labels, ids = label_question(answers, docs, stats, unresolved)
    assert labels == ["x.md", "y/index.md"]
    assert ids == [1, 3]
    assert unresolved == Counter({"nope": 1})
    assert stats["links_excluded"] == 2


def test_split_deterministic_and_stable():
    ids = range(1000, 3000)
    a = {i: split_of(i, 0, 0.4) for i in ids}
    assert a == {i: split_of(i, 0, 0.4) for i in ids}
    assert 0.35 < sum(v == "dev" for v in a.values()) / len(a) < 0.45
    # adding questions never changes existing assignments
    assert all(split_of(i, 0, 0.4) == a[i] for i in list(ids)[:100])
    assert {i: split_of(i, 1, 0.4) for i in ids} != a


def test_build_items_and_schema_loads(docs: Path, tmp_path: Path):
    q = {
        "question_id": 7,
        "title": "How &amp; why?",
        "link": "https://stackoverflow.com/q/7",
        "score": 10,
        "creation_date": 1_600_000_000,
        "owner": {"display_name": "A &amp; B", "link": "https://stackoverflow.com/u/1"},
        "accepted_answer_id": 1,
    }
    q2 = {**q, "question_id": 8}
    answers = {7: [_ans(1, "https://fastapi.tiangolo.com/x/", accepted=True)], 8: []}
    items, _, _ = build_items([q, q2], answers, docs, seed=0, dev_fraction=0.5)
    assert [i["id"] for i in items] == ["so_7"]
    assert items[0]["query"] == "How & why?"
    assert items[0]["author_display_name"] == "A & B"
    assert items[0]["license"] == "CC BY-SA 4.0"

    entry = {k: items[0][k] for k in ("id", "query", "expected_sources")}
    entry["category"] = "stackoverflow"
    path = tmp_path / "dev.json"
    path.write_text(json.dumps({"entries": [entry]}))
    ds = load_golden_dataset(path)
    assert ds.entries[0].expected_sources == ["x.md"]
    assert ds.entries[0].reference_answer == ""


def test_pre_4_0_question_rejected(docs: Path):
    q = {
        "question_id": 1,
        "title": "t",
        "link": "l",
        "score": 1,
        "creation_date": 1_500_000_000,
        "owner": {},
    }
    with pytest.raises(AssertionError):
        build_items([q], {}, docs, 0, 0.4)


def test_html_entities_in_body_do_not_leak_into_paths(docs: Path):
    body = "<a href=&quot;https://fastapi.tiangolo.com/x/&quot;>x</a>"
    stats: Counter = Counter()
    labels, _ = label_question([_ans(1, body, score=1)], docs, stats, Counter())
    assert labels == ["x.md"]
