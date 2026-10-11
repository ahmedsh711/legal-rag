"""Article record: the contract between ingestion and the rest of the system."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1"  # bump when fields change; written to articles.meta.json


class Article(BaseModel):
    """One article of the Egyptian Civil Code.

    الكتاب -> book, الباب -> chapter, الفصل -> section, الفرع -> subsection, numbered headings -> topic.
    ``id``, ``citation`` and ``citation_ar`` are derived from ``article_number`` and checked on load.
    """

    model_config = ConfigDict(extra="forbid")  # unknown field = schema mismatch

    article_number: int = Field(ge=1)
    id: str = ""
    book: str = ""
    chapter: str = ""
    section: str = ""
    subsection: str = ""
    topic: str = ""
    heading_en: str = ""
    text_ar: str = ""
    text_en: str = ""
    is_repealed: bool = False
    note: str = ""
    quality_flags: list[str] = Field(default_factory=list)  # documented source defects
    source_page: int = Field(ge=1)
    citation: str = ""
    citation_ar: str = ""

    @model_validator(mode="after")
    def _derived_fields(self) -> Article:
        n = self.article_number
        expected = {
            "id": f"eg-civil-{n}",
            "citation": f"Egyptian Civil Code, Article {n}",
            "citation_ar": f"القانون المدني المصري، المادة {n}",
        }
        for field, value in expected.items():
            current = getattr(self, field)
            if not current:
                setattr(self, field, value)
            elif current != value:
                raise ValueError(f"{field}={current!r} does not match article_number {n}")
        return self
