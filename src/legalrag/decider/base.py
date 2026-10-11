"""Decider interface: per-article relevance (rerank) and an answerability score (gate)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from legalrag.retrieval import Chunk


class DeciderUnavailableError(RuntimeError):
    """Decider call failed (timeout, 429, 5xx, bad response); callers degrade instead of failing."""


@dataclass(frozen=True)
class Decision:
    relevance: dict[int, float]  # article number -> 0..1
    answerable: float  # 0..1, compared against the gate threshold
    decider: str
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    model: str = ""


class Decider(ABC):
    name: str

    @abstractmethod
    async def decide(self, question: str, chunks: Sequence[Chunk]) -> Decision: ...


def passage_text(c: Chunk, lang: str = "en") -> str:
    """Article number and text in the preferred language, or the repeal note."""
    if c.is_repealed:
        return f"Article {c.article_number} is repealed. {c.note}"
    preferred, other = (c.text_ar, c.text_en) if lang == "ar" else (c.text_en, c.text_ar)
    return f"Article {c.article_number}: {preferred or other}"
