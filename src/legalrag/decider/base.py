"""What every decider promises, so the pipeline can swap them (the course's "model behind an
abstract base class" idea, applied to RAG decisions).

One call per question returns a ``Decision``:
- ``relevance``: article number -> 0..1, how well that article answers the question (rerank);
- ``answerable``: 0..1, can the question be answered from these articles at all (the gate).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from legalrag.retrieval import Chunk


class DeciderUnavailableError(RuntimeError):
    """The decision service failed (timeout, 429, 5xx, bad response). Callers degrade, not crash."""


@dataclass(frozen=True)
class Decision:
    relevance: dict[int, float]
    answerable: float
    decider: str
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    model: str = ""


class Decider(ABC):
    name: str

    @abstractmethod
    async def decide(self, question: str, chunks: Sequence[Chunk]) -> Decision: ...


def passage_text(c: Chunk, lang: str = "en") -> str:
    """The article as a decision model reads it: number + text in one language (or the note)."""
    if c.is_repealed:
        return f"Article {c.article_number} is repealed. {c.note}"
    preferred, other = (c.text_ar, c.text_en) if lang == "ar" else (c.text_en, c.text_ar)
    return f"Article {c.article_number}: {preferred or other}"
