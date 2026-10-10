"""Online evaluation: sample real generations from Langfuse, judge them, write scores back."""

from __future__ import annotations

import json

from legalrag.generation import REFUSAL_EN, build_messages
from legalrag.monitoring.online_eval import parse_generation, run_online_eval, sample
from tests.unit.test_generation_pipeline import chunk


def generation(i: int, answer: str = "Yes [Art. 147].", question: str = "Is a contract binding?"):
    messages = build_messages(question, [chunk(147), chunk(148)])
    return {
        "id": f"obs-{i}",
        "traceId": f"trace-{i}",
        "input": json.dumps(messages),
        "output": answer,
    }


def test_a_generation_gives_back_question_articles_and_answer():
    item = parse_generation(generation(1))
    assert item.question == "Is a contract binding?" and item.answer == "Yes [Art. 147]."
    assert len(item.contexts) == 2 and item.contexts[0].startswith("[Art. 147]")


def test_refusals_and_unreadable_inputs_are_skipped():
    assert parse_generation(generation(2, answer=REFUSAL_EN)) is None
    assert parse_generation({"id": "x", "traceId": "t", "input": "not json", "output": "a"}) is None


def test_sampling_is_a_stable_fraction():
    gens = [generation(i) for i in range(200)]
    picked = sample(gens, rate=0.05, seed=0)
    assert len(picked) == 10 and picked == sample(gens, rate=0.05, seed=0)


class FakeMetric:
    async def ascore(self, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(value=0.75 if "148" in kwargs["response"] else 1.0)


class FakeLangfuse:
    def __init__(self):
        self.scores = []

    def create_score(self, **kw):
        self.scores.append(kw)

    def flush(self):
        pass


async def test_scores_go_back_to_the_traces_and_the_mean_to_prometheus():
    gens = [
        generation(1),
        generation(2, answer="Partly [Art. 148]."),
        generation(3, answer=REFUSAL_EN),
    ]
    client = FakeLangfuse()
    summary, lines = await run_online_eval(gens, {"faithfulness": FakeMetric()}, client, rate=1.0)
    assert summary == {"sampled": 2, "scored": 2, "faithfulness": 0.875}
    assert {(s["trace_id"], s["value"]) for s in client.scores} == {
        ("trace-1", 1.0),
        ("trace-2", 0.75),
    }
    assert all(
        s["name"] == "faithfulness" and s["observation_id"].startswith("obs-")
        for s in client.scores
    )
    assert "rag_eval_faithfulness 0.875" in lines and "rag_eval_samples 2" in lines
