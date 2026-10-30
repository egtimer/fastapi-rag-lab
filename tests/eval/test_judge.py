"""Offline tests for the judge calibration pipeline (fake SDK, no API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from fastapi_rag_lab.eval.dataset import GoldenEntry
from fastapi_rag_lab.eval.judge import agreement, labels, run_judge
from fastapi_rag_lab.eval.judge.answers import sample_entries
from fastapi_rag_lab.eval.judge.judge import judge_user, parse_verdict
from fastapi_rag_lab.eval.judge.llm import MODELS, BudgetedClient, BudgetExceeded

if TYPE_CHECKING:
    from pathlib import Path


class FakeSDK:
    """Answers every call with ``reply`` and fixed token counts."""

    def __init__(self, reply: str, input_tokens: int = 3000, output_tokens: int = 200):
        self.reply = reply
        self.usage = SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply)],
            usage=self.usage,
            stop_reason="end_turn",
        )


def _items(tmp_path: Path, n: int = 3) -> Path:
    run = tmp_path / "items"
    run.mkdir()
    rows = [
        {
            "item_id": f"j-{i:02d}",
            "question_id": f"so_{i}",
            "question": f"question {i}",
            "sources": [
                {"source_file": "a.md", "heading_path": ["A"], "parent_text": "x"}
            ],
            "answer": f"answer {i}",
        }
        for i in range(1, n + 1)
    ]
    (run / "items.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run


def test_parse_verdict_reads_fenced_json_and_flags_garbage():
    v = parse_verdict('```json\n{"rationale": "ok", "label": "Supported"}\n```')
    assert (v.label, v.rationale) == ("supported", "ok")
    assert parse_verdict("no json here").label == "unparsed"
    assert parse_verdict('{"label": "maybe"}').label == "unparsed"


def test_judge_user_numbers_sources_in_order():
    text = judge_user("Q?", ["first", "second"], "A")
    assert text.index("[Source 1]\nfirst") < text.index("[Source 2]\nsecond")
    assert text.endswith("Answer:\nA")


def test_client_costs_calls_and_stops_at_the_budget(tmp_path: Path):
    model = MODELS["haiku"]
    client = BudgetedClient(
        model,
        budget_usd=0.0005,
        log_path=tmp_path / "calls.jsonl",
        sdk_client=FakeSDK("{}"),
    )
    _, usage = client.complete("s", "u", item_id="j-01")
    assert usage.cost_usd == pytest.approx((3000 * 0.10 + 200 * 0.50) / 1e6)
    client.complete("s", "u", item_id="j-02")
    with pytest.raises(BudgetExceeded):
        client.complete("s", "u", item_id="j-03")
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 2


def test_run_judge_writes_judgements_and_stops_at_the_budget(tmp_path: Path):
    items = _items(tmp_path, n=5)
    sdk = FakeSDK('{"rationale": "r", "label": "unsupported"}')
    argv = ["--items", str(items), "--model", "haiku", "--budget-usd", "0.0008"]
    argv += ["--run-id", "j", "--out-root", str(tmp_path)]
    assert run_judge.main(argv, sdk_client=sdk) == 1  # stopped by the budget
    summary = json.loads((tmp_path / "j" / "summary.json").read_text())
    assert summary["n_judged"] == 2 and summary["labels"] == {"unsupported": 2}
    assert summary["stopped"]
    assert sdk.calls[0]["output_config"] == {"effort": "low"}
    assert run_judge.main(argv, sdk_client=sdk) == 2  # run id must be new


def test_sheet_has_no_verdicts_and_labels_are_validated(tmp_path: Path):
    items = _items(tmp_path)
    assert (
        labels.main(["--items", str(items), "--questions", str(tmp_path / "none")]) == 0
    )
    sheet = (items / "labeling_sheet_01.md").read_text()
    assert "## j-01: question 1" in sheet and "answer 3" in sheet
    assert (
        labels.main(["--items", str(items), "--questions", str(tmp_path / "none")]) == 2
    )
    ids = {"j-01", "j-02", "j-03"}
    csv_path = items / "labels.csv"
    csv_path.write_text(
        "item_id,label,note\nj-01,supported,\nj-02,unsure,x\nj-03,Unsupported,\n"
    )
    assert labels.read_labels(csv_path, ids)["j-03"] == "unsupported"
    csv_path.write_text("item_id,label,note\nj-01,supported,\n")
    with pytest.raises(ValueError, match="no label"):
        labels.read_labels(csv_path, ids)
    csv_path.write_text(
        "item_id,label,note\nj-01,yes,\nj-02,supported,\nj-03,supported,\n"
    )
    with pytest.raises(ValueError, match="not in"):
        labels.read_labels(csv_path, ids)


def test_kappa_and_comparisons():
    a = ["supported", "supported", "unsupported", "unsupported"]
    b = ["supported", "unsupported", "unsupported", "unsupported"]
    # po = 0.75, pe = 0.5*0.25 + 0.5*0.75 = 0.5
    assert agreement.cohens_kappa(a, b) == pytest.approx(0.5)
    assert agreement.cohens_kappa(["supported"] * 3, ["supported"] * 3) is None
    unsupported_recall = agreement.class_agreement("unsupported")
    assert unsupported_recall(a, b) == 1.0  # both unsupported items kept
    assert agreement.class_agreement("supported")(a, b) == 0.5
    assert unsupported_recall(["supported"], ["supported"]) is None
    human = {"j-01": "supported", "j-02": "unsure", "j-03": "unsupported"}
    judge = {"j-01": "supported", "j-02": "supported", "j-03": "unparsed"}
    report = agreement.agreement_report(human, {"haiku": judge}, resamples=200)
    c = report["comparisons"]["human_vs_haiku"]
    assert c["n"] == 1 and c["accuracy"]["value"] == 1.0


def test_sample_is_seeded_and_independent_of_file_order():
    entries = [
        GoldenEntry(id=f"so_{i}", query=f"q{i}", category="stackoverflow")
        for i in range(20)
    ]
    a = [e.id for e in sample_entries(entries, 5, seed=0)]
    b = [e.id for e in sample_entries(list(reversed(entries)), 5, seed=0)]
    assert a == b and len(set(a)) == 5


def test_unparsed_keeps_the_whole_reply():
    long = (
        '{"rationale": "quotes "inside" break it", "label": "unsupported"}' + "x" * 500
    )
    v = parse_verdict(long)
    assert v.label == "unparsed"
    assert v.rationale == long  # not truncated to 200 characters
    assert v.raw == long


def test_prompt_v2_adds_the_citation_rule_and_v1_is_frozen():
    from fastapi_rag_lab.eval.judge.judge import JUDGE_SYSTEM, JUDGE_SYSTEM_V2, PROMPTS

    assert PROMPTS == {"v1": JUDGE_SYSTEM, "v2": JUDGE_SYSTEM_V2}
    assert "citation" not in JUDGE_SYSTEM
    assert "invented quotation" in JUDGE_SYSTEM_V2
    assert JUDGE_SYSTEM_V2.startswith(JUDGE_SYSTEM[:200])  # same opening, v2 extends


class FakeToolSDK(FakeSDK):
    """Answers a forced tool call with a verdict object."""

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(
            type="tool_use",
            name="verdict",
            input={"rationale": "r", "label": "supported"},
        )
        return SimpleNamespace(
            content=[block], usage=self.usage, stop_reason="tool_use"
        )


def test_structured_run_parses_the_tool_call_and_logs_text(tmp_path: Path):
    items = _items(tmp_path)
    sdk = FakeToolSDK("")
    argv = [
        "--items",
        str(items),
        "--model",
        "haiku",
        "--budget-usd",
        "1",
        "--run-id",
        "r",
        "--out-root",
        str(tmp_path),
        "--prompt",
        "v2",
        "--effort",
        "medium",
        "--structured",
        "--preregistered-labels",
        str(items / "items.jsonl"),
    ]
    assert run_judge.main(argv, sdk_client=sdk) == 0
    call = sdk.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "verdict"}
    assert call["output_config"] == {"effort": "medium"}
    assert "invented quotation" in call["system"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "r" / "judgements.jsonl").read_text().splitlines()
    ]
    assert {r["label"] for r in rows} == {"supported"}
    assert "raw_reply" not in rows[0]
    calls = [
        json.loads(line)
        for line in (tmp_path / "r" / "llm_calls.jsonl").read_text().splitlines()
    ]
    assert json.loads(calls[0]["text"])["label"] == "supported"
    summary = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert summary["judge_prompt_version"] == "v2"
    assert summary["structured_output"] is True
    assert summary["model"]["effort"] == "medium"
    assert len(summary["items_sha256"]) == 64
    assert summary["preregistered_labels"]["sha256"] == summary["items_sha256"]


def test_second_labeller_and_hash(tmp_path: Path):
    items = _items(tmp_path)
    assert labels.main(["--items", str(items)]) == 0
    assert labels.main(["--items", str(items), "--labeller", "ana"]) == 0
    assert (items / "labels_ana.csv").exists()
    rule = (items / "LABELLING_RULE.md").read_text()
    assert "invented quotation" in rule
    assert "invented quotation" not in (items / "labeling_sheet_01.md").read_text()
    assert (
        labels.main(["--items", str(items), "--labeller", "ana"]) == 2
    )  # no overwrite
    assert labels.main(["--hash", str(items / "labels_ana.csv")]) == 0
    assert (items / "labels_ana.csv.sha256").read_text().endswith("labels_ana.csv\n")


def test_agreement_with_a_second_human():
    human = {"j-01": "supported", "j-02": "unsupported", "j-03": "unsupported"}
    ana = {"j-01": "supported", "j-02": "supported", "j-03": "unsupported"}
    judge = {"j-01": "supported", "j-02": "unsupported", "j-03": "unsupported"}
    report = agreement.agreement_report(
        human,
        {"sonnet": judge, "sonnet_retest": judge},
        humans={"ana": ana},
        resamples=50,
    )
    keys = set(report["comparisons"])
    assert keys == {
        "human_vs_ana",
        "human_vs_sonnet",
        "human_vs_sonnet_retest",
        "ana_vs_sonnet",
        "ana_vs_sonnet_retest",
        "sonnet_vs_sonnet_retest",
    }
    assert report["comparisons"]["human_vs_ana"]["n"] == 3
    assert report["comparisons"]["sonnet_vs_sonnet_retest"]["accuracy"]["value"] == 1.0
