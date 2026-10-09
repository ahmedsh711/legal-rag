"""Typed access to the ``corpus`` section of params.yaml (shared by the parse and validate stages).

DVC tracks params.yaml, so every threshold here is versioned and changing one re-runs only the
stages that use it. Loading goes through Pydantic so a typo fails with a clear message instead of
a KeyError deep inside a stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Names of the validation checks. A known anomaly may allow some of them for one article.
CheckCode = Literal[
    "empty_ar", "empty_en", "arabic_in_en", "latin_in_ar", "too_long", "length_ratio"
]


class CorpusInputError(Exception):
    """Bad or missing input file for a corpus stage (params, PDF, articles.json)."""


class Anomaly(BaseModel):
    """A defect in the *source PDF* that we document instead of hiding."""

    model_config = ConfigDict(extra="forbid")
    reason: str
    flag: str  # copied into Article.quality_flags so retrieval and answers can see it
    allow: list[CheckCode] = Field(default_factory=list)  # checks that become warnings


class CorpusParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_articles: int = Field(gt=0)
    max_chars_ar: int = Field(gt=0)
    max_chars_en: int = Field(gt=0)
    length_ratio: tuple[float, float]  # allowed English/Arabic character ratio (min, max)
    book_starts: dict[int, str] = Field(default_factory=dict)  # first article of each book
    pinned: dict[int, str] = Field(default_factory=dict)  # golden Arabic phrases per article
    known_anomalies: dict[int, Anomaly] = Field(default_factory=dict)


def load_params(path: str | Path) -> CorpusParams:
    path = Path(path)
    if not path.is_file():
        raise CorpusInputError(f"params file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return CorpusParams(**data["corpus"])
    except KeyError as exc:
        raise CorpusInputError(f"{path} has no 'corpus' section") from exc
    except (yaml.YAMLError, ValidationError, TypeError) as exc:
        raise CorpusInputError(f"invalid corpus params in {path}: {exc}") from exc
