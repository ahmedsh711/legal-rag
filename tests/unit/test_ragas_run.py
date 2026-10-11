"""Which predictions get which RAGAS metric, and how scores are summarised (fake judge)."""

import math
from types import SimpleNamespace

import pytest

from legalrag.eval.metrics import Prediction
from legalrag.eval.ragas_run import score_predictions, summarize_ragas


class FakeMetric:
    def __init__(self, value):
        self.value, self.calls = value, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


def pred(**kw) -> Prediction:
    base = {
        "id": "p1-en",
        "lang": "en",
        "category": "in_scope",
        "question": "What is a lease?",
        "gold_articles": [558],
        "answer": "A lease ... [Art. 558].",
        "refused": False,
        "cited": [558],
        "context_articles": [558],
        "context_texts": ["[Art. 558] lease"],
        "reference": "A lease is ... [Art. 558].",
    }
    return Prediction(**{**base, **kw})


def metrics(faith=1.0):
    return {
        "faithfulness": FakeMetric(faith),
        "answer_relevancy": FakeMetric(0.9),
        "context_precision": FakeMetric(0.8),
        "context_recall": FakeMetric(0.7),
    }


async def test_answered_item_gets_all_four_metrics_with_the_right_inputs():
    m = metrics()
    rows = await score_predictions([pred()], m)
    assert rows[0] == {
        "id": "p1-en",
        "lang": "en",
        "faithfulness": 1.0,
        "answer_relevancy": 0.9,
        "context_precision": 0.8,
        "context_recall": 0.7,
    }
    assert m["faithfulness"].calls[0]["retrieved_contexts"] == ["[Art. 558] lease"]
    assert m["context_recall"].calls[0]["reference"] == "A lease is ... [Art. 558]."


async def test_refused_and_unanswerable_items_are_scored_only_where_meaningful():
    m = metrics()
    rows = await score_predictions(
        [pred(id="r", refused=True), pred(id="o", category="off_topic", gold_articles=[])], m
    )
    assert set(rows[0]) == {"id", "lang", "context_precision", "context_recall"}
    assert set(rows[1]) == {"id", "lang"}


class FailingMetric:
    async def ascore(self, **kwargs):
        raise RuntimeError("402 credits")


async def test_a_failing_judge_call_is_recorded_not_fatal():
    m = {**metrics(), "faithfulness": FailingMetric()}
    rows = await score_predictions([pred(), pred(id="p2-en")], m)
    assert all(r["errors"] == ["faithfulness: RuntimeError"] for r in rows)
    assert all(r["context_recall"] == 0.7 for r in rows)  # the other metrics still scored
    assert summarize_ragas(rows)["all"]["ragas_errors"] == 2


async def test_nan_scores_become_none_and_are_left_out_of_means():
    rows = await score_predictions(
        [pred(), pred(id="p1-ar", lang="ar", question="ما هو الإيجار؟")], metrics(faith=math.nan)
    )
    s = summarize_ragas(rows + [{"id": "x", "lang": "ar", "faithfulness": 0.5}])
    assert s["all"]["faithfulness"] == 0.5 and s["ar"]["faithfulness"] == 0.5
    assert s["en"]["faithfulness"] is None and s["en"]["context_recall"] == pytest.approx(0.7)
