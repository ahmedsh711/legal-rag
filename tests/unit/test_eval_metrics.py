"""Deterministic evaluation metrics over pipeline predictions (no LLM judge involved)."""

import pytest

from legalrag.eval.metrics import Prediction, gate_sweep, summarize


def pred(**kw) -> Prediction:
    base = {
        "id": "p1-en",
        "lang": "en",
        "category": "in_scope",
        "question": "What is a lease?",
        "gold_articles": [558],
        "answer": "A lease is ... [Art. 558].",
        "refused": False,
        "cited": [558],
        "context_articles": [558, 559, 560],
        "invalid_citations": [],
        "latency_ms": 1000.0,
        "prompt_tokens": 1000,
        "completion_tokens": 50,
    }
    return Prediction(**{**base, **kw})


def test_retrieval_and_citation_metrics_on_answerable_items():
    preds = [
        pred(),  # gold first in context, cited
        pred(id="p2-en", context_articles=[1, 2, 558], cited=[2]),  # gold 3rd, wrong citation
        pred(id="p3-en", context_articles=[1, 2, 3], cited=[1]),  # gold not retrieved
    ]
    m = summarize(preds)["all"]
    assert m["hit_at_1"] == pytest.approx(1 / 3)
    assert m["hit_at_5"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert m["citation_recall"] == pytest.approx(1 / 3)
    assert m["citation_precision"] == pytest.approx(1 / 3)


def test_refusal_metrics_split_answerable_and_unanswerable():
    preds = [
        pred(),
        pred(id="p2-en", refused=True, cited=[]),  # false refusal
        pred(id="p3-en", category="off_topic", gold_articles=[], refused=True, cited=[]),
        pred(id="p4-en", category="injection", gold_articles=[], refused=False, cited=[]),
    ]
    m = summarize(preds)["all"]
    assert m["false_refusal_rate"] == pytest.approx(1 / 2)
    assert m["correct_refusal_rate"] == pytest.approx(1 / 2)


def test_language_match_latency_tokens_and_cost():
    preds = [
        pred(latency_ms=1000.0),
        pred(
            id="p1-ar",
            lang="ar",
            question="ما هو الإيجار؟",
            answer="Lease [Art. 558].",
            latency_ms=3000.0,
        ),
    ]
    s = summarize(preds, price_in_per_m=0.09, price_out_per_m=0.55)
    assert s["all"]["language_match"] == pytest.approx(0.5)
    assert s["ar"]["language_match"] == 0.0 and s["en"]["language_match"] == 1.0
    assert s["all"]["latency_p50_ms"] == pytest.approx(2000.0)
    assert s["all"]["cost_usd"] == pytest.approx(2 * (1000 * 0.09 + 50 * 0.55) / 1e6)
    assert s["all"]["n"] == 2 and s["ar"]["n"] == 1


def test_gate_sweep_replays_refusals_at_each_threshold():
    preds = [
        pred(id="a1", answerable_score=0.9),
        pred(id="a2", answerable_score=0.6),
        pred(id="a3", answerable_score=0.6, refused=True),  # the LLM refused anyway
        pred(id="o1", category="off_topic", gold_articles=[], answerable_score=0.7),
        pred(id="o2", category="injection", gold_articles=[], answerable_score=0.1),
    ]
    rows = {r["threshold"]: r for r in gate_sweep(preds, [0.0, 0.65, 0.8])}
    assert rows[0.0]["false_refusal_rate"] == pytest.approx(1 / 3)  # only the LLM's own refusal
    assert rows[0.0]["correct_refusal_rate"] == 0.0
    assert rows[0.65]["false_refusal_rate"] == pytest.approx(2 / 3)
    assert rows[0.65]["correct_refusal_rate"] == pytest.approx(1 / 2)
    assert rows[0.8]["correct_refusal_rate"] == 1.0


def test_gate_sweep_ignores_the_runs_own_gate_refusals():
    # gated during the run at 0.75: below that threshold the sweep must not count it as refused
    p = pred(id="g", answerable_score=0.5, refused=True, gated=True)
    rows = {r["threshold"]: r for r in gate_sweep([p], [0.3, 0.6])}
    assert rows[0.3]["false_refusal_rate"] == 0.0 and rows[0.6]["false_refusal_rate"] == 1.0


def test_repealed_items_count_as_answerable():
    p = pred(category="repealed", gold_articles=[60], cited=[60], context_articles=[60])
    m = summarize([p])["all"]
    assert m["hit_at_1"] == 1.0 and m["citation_recall"] == 1.0 and m["false_refusal_rate"] == 0.0
