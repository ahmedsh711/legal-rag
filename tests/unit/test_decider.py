"""Deciders (Jev, local cross-encoder) and how the pipeline uses their decisions."""

import json

import httpx
import pytest

from legalrag.decider.base import DeciderUnavailableError, Decision
from legalrag.decider.jev import JevDecider
from legalrag.decider.local import LocalDecider
from legalrag.generation import REFUSAL_EN, Generator
from legalrag.pipeline import RagPipeline
from legalrag.retrieval import Chunk
from tests.fakes import FakeLLM


def chunk(n: int, **kw) -> Chunk:
    return Chunk(article_number=n, citation=f"Egyptian Civil Code, Article {n}",
                 citation_ar=f"القانون المدني المصري، المادة {n}",
                 text_ar=kw.pop("text_ar", f"نص {n}"), text_en=kw.pop("text_en", f"Text {n}."), **kw)  # fmt: skip


def jev_reply(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    answers = {
        qid: {"type": "score", "score": 3.0 if qid == "rel_558" else 0.3} for qid in body["questions"]
        if qid.startswith("rel_")
    }  # fmt: skip
    answers["answerable"] = {"type": "noul", "noul": 0.9}
    return httpx.Response(200, json={"model": "typesafe/jev-1.13-x", "answers": answers,
                                     "usage": {"input_tokens": 500, "cost": 2e-5}})  # fmt: skip


async def test_jev_sends_english_passages_and_parses_scores():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return jev_reply(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    jev = JevDecider(client, base_url="https://openrouter.ai/api/v1", api_key="k", model="jev-1.13")
    d = await jev.decide("ما هو الإيجار؟", [chunk(147), chunk(558, text_en="A lease is ...")])
    assert seen["url"] == "https://openrouter.ai/api/v1/systemone" and seen["auth"] == "Bearer k"
    assert seen["body"]["state"]["query"] == "ما هو الإيجار؟"
    assert "A lease is ..." in seen["body"]["state"]["passages"]["p558"]  # English text for Jev
    assert seen["body"]["questions"]["answerable"]["type"] == "noul"
    assert d.relevance == {147: pytest.approx(0.1), 558: pytest.approx(1.0)}  # score / 3
    assert d.answerable == 0.9 and d.decider == "jev" and d.cost_usd == 2e-5


@pytest.mark.parametrize("status", [429, 500, 529])
async def test_jev_errors_become_decider_unavailable(status):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(status)))
    jev = JevDecider(client, base_url="https://x/api/v1", api_key="k", model="jev-1.13")
    with pytest.raises(DeciderUnavailableError):
        await jev.decide("q?", [chunk(1)])


async def test_local_decider_uses_cross_encoder_scores():
    local = LocalDecider(score_pairs=lambda q, texts: [0.0 if "147" in t else 3.0 for t in texts])
    d = await local.decide("What is a lease?", [chunk(147), chunk(558)])
    assert d.relevance[558] > 0.9 and d.relevance[147] == pytest.approx(0.5)  # sigmoid
    assert d.answerable == d.relevance[558] and d.decider == "local"


class FixedDecider:
    name = "fixed"

    def __init__(self, decision=None, error=False):
        self.decision, self.error, self.calls = decision, error, 0

    async def decide(self, question, chunks):
        self.calls += 1
        if self.error:
            raise DeciderUnavailableError("down")
        return self.decision


class StubRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question, top_n=None, book=None):
        return self.chunks


def make_pipe(decider, reply="Lease [Art. 558]."):
    llm = FakeLLM(reply)
    pipe = RagPipeline(StubRetriever([chunk(147), chunk(374), chunk(558)]),
                       Generator(llm, "m", 100, 0.0), context_size=2, decider=decider,
                       gate_threshold=0.5)  # fmt: skip
    return pipe, llm


async def test_decider_reorders_the_context():
    d = Decision(relevance={147: 0.1, 374: 0.2, 558: 0.9}, answerable=0.9, decider="fixed")
    pipe, _ = make_pipe(FixedDecider(d))
    ans = await pipe.ask("What is a lease?")
    assert [c.article_number for c in ans.context] == [558, 374]
    assert ans.decision is d


async def test_gate_refuses_without_calling_the_llm():
    d = Decision(relevance={147: 0.1, 374: 0.1, 558: 0.1}, answerable=0.2, decider="fixed")
    pipe, llm = make_pipe(FixedDecider(d))
    ans = await pipe.ask("What is the tax rate?")
    assert ans.refused and ans.answer == REFUSAL_EN and llm.calls == []
    assert ans.guardrails == ["gate:unanswerable"]


async def test_explicit_article_questions_skip_the_gate_and_stay_first():
    d = Decision(relevance={147: 0.0, 374: 0.9, 558: 0.1}, answerable=0.0, decider="fixed")
    pipe, llm = make_pipe(FixedDecider(d), reply="Article 147 ... [Art. 147].")
    ans = await pipe.ask("What does Article 147 say?")
    assert not ans.refused and ans.context[0].article_number == 147 and llm.calls


async def test_decider_outage_falls_back_to_retrieval_order():
    pipe, llm = make_pipe(FixedDecider(error=True))
    ans = await pipe.ask("What is a lease?")
    assert [c.article_number for c in ans.context] == [147, 374] and ans.decision is None
    assert llm.calls  # still answers: degraded, not down


async def test_local_decider_errors_become_unavailable_so_the_pipeline_degrades():
    def broken(question, texts):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(DeciderUnavailableError):
        await LocalDecider(score_pairs=broken).decide("q?", [chunk(1)])


async def test_jev_bad_cost_value_does_not_lose_the_decision():
    def handler(request: httpx.Request) -> httpx.Response:
        reply = jev_reply(request)
        data = json.loads(reply.content)
        data["usage"]["cost"] = "n/a"
        return httpx.Response(200, json=data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    jev = JevDecider(client, base_url="https://x/api/v1", api_key="k", model="jev-1.13")
    d = await jev.decide("q?", [chunk(558)])
    assert d.relevance[558] == pytest.approx(1.0) and d.cost_usd == 0.0
