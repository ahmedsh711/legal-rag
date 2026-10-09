"""The RAG pipeline: retrieve -> (decide, from Phase 3) -> generate -> check citations.

Retrieval is CPU work (embedding the question) plus a Qdrant call, so it runs in a worker thread
(``asyncio.to_thread``) and never blocks the event loop that serves other users.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from legalrag.generation import (
    PROMPT_VERSION,
    Generator,
    build_messages,
    cited_articles,
    detect_language,
    refusal_for,
)
from legalrag.logging_conf import get_logger
from legalrag.retrieval import Chunk, article_numbers_in

log = get_logger(__name__)


class SupportsRetrieve(Protocol):
    def retrieve(
        self, question: str, top_n: int | None = None, book: str | None = None
    ) -> list[Chunk]: ...


@dataclass
class Answer:
    question: str
    answer: str
    language: str
    sources: list[Chunk]
    context: list[Chunk]  # everything the model was shown
    refused: bool
    invalid_citations: list[int]  # cited by the model but never retrieved: a red flag
    prompt_version: str = PROMPT_VERSION
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class RagPipeline:
    def __init__(
        self,
        retriever: SupportsRetrieve,
        generator: Generator,
        context_size: int = 5,
        top_n: int = 12,
    ):
        self.retriever, self.generator = retriever, generator
        self.context_size, self.top_n = context_size, top_n

    async def _context(self, question: str, book: str | None) -> list[Chunk]:
        chunks = await asyncio.to_thread(self.retriever.retrieve, question, self.top_n, book)
        # articles the user names come first and are always shown to the model
        return chunks[: max(self.context_size, len(article_numbers_in(question)), 1)]

    def _finish(
        self, question: str, language: str, text: str, context: list[Chunk]
    ) -> dict[str, Any]:
        cited = cited_articles(text)
        known = {c.article_number for c in context}
        refused = text.strip() == refusal_for(language) or not context
        if bad := [n for n in cited if n not in known]:
            log.warning("invalid_citations", cited=bad)
        return {
            "sources": [c for c in context if c.article_number in cited] if not refused else [],
            "invalid_citations": [n for n in cited if n not in known],
            "refused": refused,
        }

    async def ask(self, question: str, book: str | None = None) -> Answer:
        t0 = time.perf_counter()
        language = detect_language(question)
        context = await self._context(question, book)
        timings = {"retrieve": _ms(t0)}
        if not context:  # nothing to ground an answer on: refuse without spending tokens
            return Answer(
                question,
                refusal_for(language),
                language,
                [],
                [],
                True,
                [],
                timings_ms={**timings, "total": _ms(t0)},
            )

        t1 = time.perf_counter()
        completion = await self.generator.complete(build_messages(question, context))
        timings["generate"] = _ms(t1)
        timings["total"] = _ms(t0)
        checked = self._finish(question, language, completion.text, context)
        log.info(
            "answered",
            language=language,
            context=len(context),
            refused=checked["refused"],
            **timings,
        )
        return Answer(
            question=question,
            answer=completion.text,
            language=language,
            context=context,
            model=completion.model,
            usage={
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
            },
            timings_ms=timings,
            **checked,
        )

    async def ask_stream(
        self, question: str, book: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Server-sent events: {"type": "token", "text": ...} ... then one {"type": "done", ...}."""
        t0 = time.perf_counter()
        language = detect_language(question)
        context = await self._context(question, book)
        if not context:
            yield {"type": "token", "text": refusal_for(language)}
            yield {"type": "done", "refused": True, "sources": [], "invalid_citations": []}
            return
        stream = self.generator.stream(build_messages(question, context))
        parts: list[str] = []
        first_token_ms = None
        async for token in stream.tokens():
            if first_token_ms is None:
                first_token_ms = _ms(t0)  # time to first token, what the user feels as speed
            parts.append(token)
            yield {"type": "token", "text": token}
        checked = self._finish(question, language, "".join(parts), context)
        yield {
            "type": "done",
            "refused": checked["refused"],
            "sources": [
                {
                    "article_number": c.article_number,
                    "citation": c.citation,
                    "citation_ar": c.citation_ar,
                }
                for c in checked["sources"]
            ],
            "invalid_citations": checked["invalid_citations"],
            "usage": {
                "prompt_tokens": stream.prompt_tokens,
                "completion_tokens": stream.completion_tokens,
            },
            "timings_ms": {"ttft": first_token_ms, "total": _ms(t0)},
            "prompt_version": PROMPT_VERSION,
        }
