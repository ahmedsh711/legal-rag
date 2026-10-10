"""Jev (TypeSafe's decision model) through OpenRouter's ``/systemone`` endpoint.

One request per question asks one ``score`` question per article ("how well does it answer?") and
one ``noul`` (a yes/no probability: "can the question be answered from these articles?"). Jev
returns probabilities, not text, so there is nothing to parse and nothing to hallucinate.

Articles are sent in English: Jev's documentation says English is its primary language and other
languages are "handled but not equally well". The question is sent as the user typed it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx

from legalrag.decider.base import Decider, DeciderUnavailableError, Decision, passage_text
from legalrag.logging_conf import get_logger
from legalrag.retrieval import Chunk

log = get_logger(__name__)

LEVELS = ["Irrelevant", "Related but does not answer", "Partly answers", "Directly answers"]
ANSWERABLE = {
    "type": "noul",
    "instructions": "Can `query` be answered using only the articles in `passages`?",
    "criteria": {
        "true": "the passages state the rule that answers the query",
        "false": "the answer is missing from the passages, or the query is not a legal question",
    },
}


class JevDecider(Decider):
    name = "jev"

    def __init__(self, client: httpx.AsyncClient, base_url: str, api_key: str, model: str):
        self.client, self.url = client, base_url.rstrip("/") + "/systemone"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.model = model

    def _body(self, question: str, chunks: Sequence[Chunk]) -> dict[str, Any]:
        questions: dict[str, Any] = {
            f"rel_{c.article_number}": {
                "type": "score",
                "instructions": f"How well does `passages.p{c.article_number}` answer `query`?",
                "criteria": LEVELS,
            }
            for c in chunks
        }
        questions["answerable"] = ANSWERABLE
        passages = {f"p{c.article_number}": passage_text(c, "en") for c in chunks}
        return {
            "model": self.model,
            "state": {"query": question, "passages": passages},
            "questions": questions,
        }

    async def decide(self, question: str, chunks: Sequence[Chunk]) -> Decision:
        t0 = time.perf_counter()
        try:
            r = await self.client.post(
                self.url, json=self._body(question, chunks), headers=self.headers
            )
            r.raise_for_status()
            data = r.json()
            answers = data["answers"]
            top = len(LEVELS) - 1
            relevance = {
                c.article_number: float(answers[f"rel_{c.article_number}"]["score"]) / top
                for c in chunks
            }
            answerable = float(answers["answerable"]["noul"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise DeciderUnavailableError(f"jev: {type(exc).__name__}: {exc}") from exc
        return Decision(
            relevance=relevance,
            answerable=answerable,
            decider=self.name,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            cost_usd=_cost(data),
            model=str(data.get("model", "")),
        )

    async def aclose(self) -> None:
        await self.client.aclose()


def _cost(data: dict[str, Any]) -> float:
    """OpenRouter's reported cost; a missing or odd value must not lose a good decision."""
    try:
        return float((data.get("usage") or {}).get("cost") or 0.0)
    except (TypeError, ValueError):
        return 0.0
