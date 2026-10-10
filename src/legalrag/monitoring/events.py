"""Prediction events: one JSON line per answer, the raw material for drift and log drill-down.

What is in an event: the features a drift test compares (language, question length, which
articles and which book retrieval landed on, the decider's answerability score), what happened
(refused, guards fired, stage timings) and what served it (prompt, model, index, decider).
What is never in it: the question or the answer text, the IP or the API key. The request id is
the link to the full logs of that one request when someone needs to drill down.

Files: ``data/events/events-YYYY-MM-DD.jsonl`` (one per UTC day, easy to expire).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

STAGES = ("guard", "retrieve", "decide", "generate", "ttft", "total")


class PredictionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: str
    endpoint: Literal["ask", "ask_stream", "eval"]
    lang: Literal["ar", "en"]
    question_chars: int
    question_words: int
    pii_redacted: bool
    top_articles: list[int]  # what the model was shown, best first
    top_book: str  # book of the first article: the "topic" of the question, as retrieval saw it
    answerable_score: float | None  # the decider's gate score, when a decider ran
    refused: bool
    guardrails: list[str]
    timings_ms: dict[str, float]
    prompt_version: str
    llm_model: str
    index_collection: str
    decider: str


def _common(question: str, serving: Mapping[str, str]) -> dict[str, Any]:
    return {"question_chars": len(question), "question_words": len(question.split()),
            **{k: serving[k] for k in ("prompt_version", "llm_model", "index_collection",
                                       "decider")}}  # fmt: skip


def event_from_answer(answer: Any, question: str, request_id: str, endpoint: str,
                      serving: Mapping[str, str]) -> PredictionEvent:  # fmt: skip
    """``question`` is only measured (length), never stored."""
    context = answer.context
    return PredictionEvent(
        request_id=request_id, endpoint=endpoint, lang=answer.language,
        pii_redacted=any(g.startswith("pii:") for g in answer.guardrails),
        top_articles=[c.article_number for c in context],
        top_book=context[0].book if context else "",
        answerable_score=answer.decision.answerable if answer.decision else None,
        refused=answer.refused, guardrails=list(answer.guardrails),
        timings_ms={k: v for k, v in answer.timings_ms.items() if k in STAGES},
        **{**_common(question, serving), "prompt_version": answer.prompt_version},
    )  # fmt: skip


def event_from_stream_done(done: Mapping[str, Any], question: str, request_id: str,
                           serving: Mapping[str, str]) -> PredictionEvent:  # fmt: skip
    articles = done.get("context_articles", [])
    return PredictionEvent(
        request_id=request_id, endpoint="ask_stream", lang=done.get("language", "en"),
        pii_redacted=any(g.startswith("pii:") for g in done.get("guardrails", [])),
        top_articles=articles, top_book=done.get("top_book", ""),
        answerable_score=done.get("answerable_score"), refused=done.get("refused", False),
        guardrails=list(done.get("guardrails", [])),
        timings_ms={k: v for k, v in done.get("timings_ms", {}).items() if k in STAGES and v is not None},
        **{**_common(question, serving),
           "prompt_version": done.get("prompt_version", serving["prompt_version"])},
    )  # fmt: skip


def event_from_prediction(pred: Any, books: Mapping[int, str],
                          serving: Mapping[str, str]) -> PredictionEvent:  # fmt: skip
    """Reference events from an evaluation run (the traffic the system was judged on)."""
    articles = list(pred.context_articles)
    return PredictionEvent(
        request_id=pred.id, endpoint="eval", lang=pred.lang,
        pii_redacted=any(g.startswith("pii:") for g in pred.guardrails),
        top_articles=articles, top_book=books.get(articles[0], "") if articles else "",
        answerable_score=pred.answerable_score, refused=pred.refused,
        guardrails=list(pred.guardrails), timings_ms={"total": pred.latency_ms},
        **_common(pred.question, serving),
    )  # fmt: skip


class EventLog:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def write(self, event: PredictionEvent) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"events-{event.ts:%Y-%m-%d}.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(event.model_dump_json() + "\n")


def read_events(directory: str | Path, since: datetime | None = None,
                until: datetime | None = None) -> list[PredictionEvent]:  # fmt: skip
    events = []
    for path in sorted(Path(directory).glob("events-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = PredictionEvent.model_validate_json(line)
                if (since is None or event.ts >= since) and (until is None or event.ts < until):
                    events.append(event)
    return events
