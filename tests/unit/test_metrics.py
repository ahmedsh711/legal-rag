"""The Prometheus metric contract: what one answer records, and labels that cannot explode."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from legalrag.decider.base import Decision
from legalrag.observability.metrics import RagMetrics, render
from legalrag.pipeline import Answer


@pytest.fixture
def metrics():
    registry = CollectorRegistry()
    return RagMetrics(registry), registry


def answer(**kw) -> Answer:
    base = {"question": "q", "answer": "a [Art. 1]", "language": "en", "sources": [],
            "context": [], "refused": False, "invalid_citations": [],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 40},
            "timings_ms": {"guard": 0.1, "retrieve": 250.0, "decide": 400.0, "generate": 900.0,
                           "total": 1550.0}}  # fmt: skip
    return Answer(**{**base, **kw})


def value(registry, name, **labels):
    return registry.get_sample_value(name, labels or None)


def test_an_answer_records_stages_tokens_outcome_and_decision(metrics):
    m, reg = metrics
    decision = Decision(relevance={1: 0.9}, answerable=0.9, decider="jev", cost_usd=0.0001)
    m.observe_answer(answer(decision=decision), backend="gemini")
    assert value(reg, "rag_stage_seconds_count", stage="retrieve") == 1
    assert value(reg, "rag_stage_seconds_sum", stage="generate") == pytest.approx(0.9)
    assert value(reg, "rag_stage_seconds_count", stage="total") is None  # total is not a stage
    assert value(reg, "rag_llm_tokens_total", backend="gemini", type="prompt") == 1200
    assert value(reg, "rag_answers_total", outcome="answered", prompt_version="v3") == 1
    assert value(reg, "rag_decisions_total", decider="jev", outcome="pass") == 1
    assert value(reg, "rag_decider_cost_usd_total") == pytest.approx(0.0001)


def test_every_fired_guard_is_counted_and_unknown_names_cannot_add_series(metrics):
    m, reg = metrics
    m.observe_answer(answer(refused=True, gated=True,
                            guardrails=["pii:phone", "gate:unanswerable", "made-up:guard"]),
                     backend="vllm")  # fmt: skip
    assert value(reg, "rag_guardrail_total", guard="pii:phone") == 1
    assert value(reg, "rag_guardrail_total", guard="gate:unanswerable") == 1
    assert value(reg, "rag_guardrail_total", guard="other") == 1  # bounded label set
    assert value(reg, "rag_answers_total", outcome="refused", prompt_version="v3") == 1


def test_requests_are_labelled_by_route_not_by_raw_path(metrics):
    m, reg = metrics
    m.observe_request("POST", "/ask", 200, 1.2)
    m.observe_request("GET", "/docs/../../etc/passwd", 404, 0.01)  # a scanner: one bounded series
    assert value(reg, "rag_requests_total", endpoint="ask", status="200") == 1
    assert value(reg, "rag_requests_total", endpoint="other", status="404") == 1
    assert value(reg, "rag_request_seconds_count", endpoint="ask") == 1


def test_a_stream_records_time_to_first_token(metrics):
    m, reg = metrics
    done = {"type": "done", "refused": False, "guardrails": [], "usage": {"prompt_tokens": 10,
            "completion_tokens": 5}, "timings_ms": {"ttft": 310.0, "total": 900.0}}  # fmt: skip
    m.observe_stream_done(done, backend="vllm")
    assert value(reg, "rag_ttft_seconds_sum") == pytest.approx(0.31)
    assert value(reg, "rag_llm_tokens_total", backend="vllm", type="completion") == 5


def test_render_exposes_the_text_format(metrics):
    m, reg = metrics
    m.set_info(app_version="0.3.0", prompt_version="v3", llm_model="m", decider="jev",
               index_collection="articles_x", config_source="env")  # fmt: skip
    body, content_type = render(reg)
    assert content_type.startswith("text/plain") and b"rag_info{" in body


def test_answers_carry_a_bounded_prompt_version(metrics):
    m, reg = metrics
    m.observe_answer(answer(prompt_version="v4"), backend="vllm")
    m.observe_answer(answer(prompt_version="'; DROP TABLE"), backend="vllm")
    assert value(reg, "rag_answers_total", outcome="answered", prompt_version="v4") == 1
    assert value(reg, "rag_answers_total", outcome="answered", prompt_version="other") == 1


def test_a_stream_records_its_stages_and_errors_are_counted(metrics):
    m, reg = metrics
    done = {"type": "done", "refused": False, "guardrails": [], "usage": {},
            "prompt_version": "v3",
            "timings_ms": {"guard": 0.1, "retrieve": 200.0, "ttft": 300.0, "total": 900.0}}  # fmt: skip
    m.observe_stream_done(done, backend="vllm")
    assert value(reg, "rag_stage_seconds_count", stage="retrieve") == 1
    m.observe_stream_error()
    assert value(reg, "rag_answers_total", outcome="error", prompt_version="none") == 1
