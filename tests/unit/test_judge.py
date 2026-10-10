"""LLM judge: label sheet, verdict parsing, agreement with humans, bias probes (fake LLM)."""

import csv
import json
from types import SimpleNamespace

import pytest

from legalrag.eval.judge import (
    FABRICATED,
    PADDING,
    Judge,
    agreement,
    calibrate,
    cohen_kappa,
    read_labels,
    with_fabricated_claim,
    write_label_sheet,
)
from legalrag.eval.metrics import Prediction


def pred(i: int, lang: str = "en", refused: bool = False, category: str = "in_scope") -> Prediction:
    return Prediction(id=f"p{i}-{lang}", lang=lang, category=category, question=f"q{i}?",
                      gold_articles=[i] if category != "off_topic" else [], answer=f"a{i} [Art. {i}].",
                      refused=refused, cited=[i], context_articles=[i],
                      context_texts=[f"[Art. {i}] text {i}"])  # fmt: skip


def test_label_sheet_has_only_answered_items_balanced_by_language(tmp_path):
    preds = [pred(i, lang) for i in range(1, 15) for lang in ("ar", "en")]
    preds += [pred(99, refused=True), pred(98, category="off_topic")]
    path = tmp_path / "to_label.csv"
    write_label_sheet(preds, path, n=10, seed=1)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM: Excel shows Arabic correctly
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert len(rows) == 10 and sum(r["lang"] == "ar" for r in rows) == 5
    assert {"id", "question", "articles", "answer", "supported (1/0)", "note"} <= set(rows[0])
    assert not {"p99-en", "p98-en"} & {r["id"] for r in rows}


def test_read_labels_ignores_blank_rows(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("id,supported (1/0)\nA,1\nB,0\nC,\n", encoding="utf-8-sig")
    assert read_labels(path) == {"A": 1, "B": 0}


def test_agreement_and_kappa():
    human = {"a": 1, "b": 1, "c": 0, "d": 0}
    assert agreement(human, {"a": 1, "b": 1, "c": 0, "d": 1}) == pytest.approx(0.75)
    assert cohen_kappa(human, dict(human)) == pytest.approx(1.0)
    assert cohen_kappa(human, {"a": 1, "b": 0, "c": 1, "d": 0}) == pytest.approx(0.0)


class FakeChat:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


async def test_judge_parses_json_verdict_and_sends_articles():
    llm = FakeChat([json.dumps({"supported": False, "unsupported_claims": ["x"]})])
    judge = Judge(llm, "judge-model")
    verdict = await judge.supported(pred(1))
    assert verdict == 0
    sent = llm.calls[0]
    assert sent["model"] == "judge-model" and sent["response_format"] == {"type": "json_object"}
    assert "[Art. 1] text 1" in sent["messages"][1]["content"]


async def test_judge_padding_probe_appends_harmless_text():
    llm = FakeChat([json.dumps({"supported": True})])
    await Judge(llm, "m").supported(pred(1), pad=True)
    assert PADDING["en"] in llm.calls[0]["messages"][1]["content"]


def test_fabricated_claim_makes_a_known_negative():
    neg = with_fabricated_claim(pred(3, lang="ar"))
    assert neg.id == "p3-ar-neg" and neg.answer.endswith(FABRICATED["ar"])
    assert neg.context_texts == ["[Art. 3] text 3"]  # same articles: the new claim is unsupported


class ScriptedJudge:
    """Says 'supported' unless the answer contains the fabricated claim (a perfect judge)."""

    def __init__(self, model, fooled_by_padding=False):
        self.model, self.fooled = model, fooled_by_padding

    async def supported(self, p, pad=False):
        if pad and self.fooled:
            return 0
        return 0 if any(c in p.answer for c in FABRICATED.values()) else 1


async def test_calibration_adds_negatives_so_kappa_means_something():
    preds = [pred(i) for i in range(1, 5)]
    human = {p.id: 1 for p in preds}  # humans found every real answer supported
    report = await calibrate(preds, human, ScriptedJudge("judge"), same_family=None, ragas={})
    assert report["n_labelled"] == 4 and report["n_synthetic_negatives"] == 4
    assert report["judge"]["agreement"] == 1.0 and report["judge"]["kappa"] == 1.0
    assert report["self_preference_probe"] == "not measured: judge and generator share a model"
    assert report["verbosity_probe"]["verdict_flips"] == 0


async def test_verbosity_probe_counts_flips():
    preds = [pred(i) for i in range(1, 3)]
    report = await calibrate(preds, {p.id: 1 for p in preds},
                             ScriptedJudge("judge", fooled_by_padding=True), None, {})  # fmt: skip
    assert report["verbosity_probe"] == {"verdict_flips": 2, "of": 2}
