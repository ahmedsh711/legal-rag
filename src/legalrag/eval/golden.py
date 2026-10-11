"""Golden set: questions with known answers, in mirrored Arabic/English pairs.

``in_scope`` and ``explicit_ref`` items must cite ``gold_articles``; ``repealed`` items expect the
answer to say the article is repealed; ``off_topic`` and ``injection`` items expect the exact
refusal sentence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legalrag.generation import detect_language, refusal_for
from legalrag.ingest.schema import Article

Category = Literal["in_scope", "explicit_ref", "repealed", "off_topic", "injection"]
UNANSWERABLE = {"off_topic", "injection"}


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    pair: str
    lang: Literal["ar", "en"]
    category: Category
    question: str = Field(min_length=3)
    gold_articles: list[int]
    reference: str = Field(min_length=3)

    @model_validator(mode="after")
    def _consistent(self) -> GoldenItem:
        if detect_language(self.question) != self.lang:
            raise ValueError(f"{self.id}: language {self.lang!r} does not match the question")
        if self.category in UNANSWERABLE:
            if self.gold_articles:
                raise ValueError(f"{self.id}: {self.category} items have no gold articles")
            if self.reference != refusal_for(self.lang):
                raise ValueError(f"{self.id}: reference must be the exact refusal sentence")
        elif not self.gold_articles:
            raise ValueError(f"{self.id}: {self.category} items need gold articles")
        return self


def load_golden(path: str | Path) -> list[GoldenItem]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [GoldenItem(**json.loads(line)) for line in lines if line.strip()]


def check_golden(items: list[GoldenItem], corpus: dict[int, Article] | None) -> list[str]:
    """Problems across items: duplicates, unmirrored pairs and (with a corpus) wrong articles."""
    problems: list[str] = []
    seen: set[str] = set()
    pairs: dict[str, dict[str, GoldenItem]] = defaultdict(dict)
    for it in items:
        if it.id in seen:
            problems.append(f"duplicate id {it.id}")
        seen.add(it.id)
        pairs[it.pair][it.lang] = it
        if corpus is not None:
            problems += _corpus_problems(it, corpus)
    for pair, by_lang in pairs.items():
        for lang in ("ar", "en"):
            if lang not in by_lang:
                problems.append(f"{pair}: missing {lang}")
        if len(by_lang) == 2:
            ar, en = by_lang["ar"], by_lang["en"]
            if (ar.category, ar.gold_articles) != (en.category, en.gold_articles):
                problems.append(f"{pair}: AR and EN do not mirror each other")
    return problems


def _corpus_problems(it: GoldenItem, corpus: dict[int, Article]) -> list[str]:
    out = []
    for n in it.gold_articles:
        article = corpus.get(n)
        if article is None:
            out.append(f"{it.id}: article {n} is not in the corpus")
        elif it.category == "repealed" and not article.is_repealed:
            out.append(f"{it.id}: article {n} is not repealed")
        elif it.category != "repealed" and article.is_repealed:
            out.append(f"{it.id}: article {n} is repealed")
    return out
