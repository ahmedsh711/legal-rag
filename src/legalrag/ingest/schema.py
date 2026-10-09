"""The record we produce for every article: the contract between ingestion and everything else."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Article(BaseModel):
    """One article of the Egyptian Civil Code (handbook schema, extended where ingestion needed it).

    Hierarchy follows the Arabic headings, named like the English edition of the PDF:
    الكتاب -> book, الباب -> chapter, الفصل -> section, الفرع -> subsection, numbered headings -> topic.
    """

    article_number: int = Field(ge=1)
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
    source_page: int = Field(ge=1)
    citation: str = ""

    def model_post_init(self, __context) -> None:
        if not self.citation:
            self.citation = f"Egyptian Civil Code, Article {self.article_number}"
