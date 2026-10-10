"""The CI quality gate: frozen contexts, real pipeline, two thresholds, fails closed."""

from __future__ import annotations

import json

from legalrag.eval.metrics import Prediction
from legalrag.eval.smoke import FrozenRetriever, freeze, load_smoke, smoke_predictions, verdict
from legalrag.generation import Generator
from legalrag.pipeline import RagPipeline
from tests.conftest import SAMPLE_ARTICLES
from tests.fakes import FakeLLM


def pred(i: int, cited: list[int], **kw) -> Prediction:
    base = {"id": f"q{i}", "lang": "en", "category": "in_scope", "question": "q?",
            "gold_articles": [374], "answer": "a", "refused": False, "cited": cited,
            "context_articles": [374]}  # fmt: skip
    return Prediction(**{**base, **kw})


def test_gate_passes_when_both_thresholds_hold():
    preds = [pred(i, [374]) for i in range(10)]
    rows = [{"id": f"q{i}", "faithfulness": 0.9} for i in range(10)]
    v = verdict(preds, rows, min_faithfulness=0.75, min_cited=0.8, max_errors=2)
    assert v["passed"] and v["faithfulness"] == 0.9 and v["cited_gold"] == 1.0


def test_gate_names_every_threshold_it_misses():
    preds = [pred(i, [374] if i < 7 else []) for i in range(10)]
    rows = [{"id": f"q{i}", "faithfulness": 0.6} for i in range(10)]
    v = verdict(preds, rows, min_faithfulness=0.75, min_cited=0.8, max_errors=2)
    assert not v["passed"] and len(v["reasons"]) == 2  # faithfulness 0.6 and cited 0.7


def test_gate_fails_closed_when_it_cannot_measure():
    preds = [pred(i, [374], error="RateLimitError" if i < 3 else "") for i in range(10)]
    rows = [{"id": f"q{i}", "faithfulness": 1.0} for i in range(3, 10)]
    v = verdict(preds, rows, min_faithfulness=0.75, min_cited=0.8, max_errors=2)
    assert not v["passed"] and "3 items could not be measured" in v["reasons"][0]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def test_freeze_then_load_gives_the_exact_articles_back(tmp_path):
    golden = [{"id": f"p0{n}-{lang}", "pair": f"p0{n}", "lang": lang, "category": "in_scope",
               "question": f"ما حكم المادة {n}؟" if lang == "ar" else f"What about rule {n}?",
               "gold_articles": [374], "reference": "ref ok"}
              for n in (1, 2) for lang in ("ar", "en")]  # fmt: skip
    write_jsonl(tmp_path / "golden.jsonl", golden)
    write_jsonl(
        tmp_path / "preds.jsonl", [{"id": g["id"], "context_articles": [374, 147]} for g in golden]
    )
    (tmp_path / "articles.json").write_text(
        json.dumps([a.model_dump() for a in SAMPLE_ARTICLES], ensure_ascii=False), "utf-8"
    )
    rows = freeze(tmp_path / "preds.jsonl", tmp_path / "articles.json", tmp_path / "golden.jsonl",
                  pairs=1)  # fmt: skip
    write_jsonl(tmp_path / "smoke.jsonl", rows)
    items, contexts = load_smoke(tmp_path / "smoke.jsonl")
    assert [i.id for i in items] == ["p01-ar", "p01-en"]
    chunks = contexts["ما حكم المادة 1؟"]
    assert [c.article_number for c in chunks] == [374, 147]
    assert chunks[0].text_en.startswith("The term of prescription")


async def test_smoke_runs_the_real_pipeline_on_the_frozen_articles(tmp_path):
    from legalrag.eval.golden import GoldenItem
    from tests.unit.test_generation_pipeline import chunk

    contexts = {"What is the prescription period?": [chunk(374)]}
    items = [GoldenItem(id="p01-en", pair="p01", lang="en", category="in_scope",
                        question="What is the prescription period?", gold_articles=[374],
                        reference="Fifteen years [Art. 374].")]  # fmt: skip
    llm = FakeLLM("Fifteen years [Art. 374].")
    pipe = RagPipeline(FrozenRetriever(contexts), Generator(llm, "m", 100, 0.0), context_size=5)
    preds = await smoke_predictions(pipe, items)
    assert preds[0].cited == [374] and preds[0].context_texts[0].startswith("[Art. 374]")
