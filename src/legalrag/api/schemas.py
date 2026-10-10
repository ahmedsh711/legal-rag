"""API request and response models; invalid input is rejected with a 422 before any model call."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from legalrag.ingest.normalize import has_arabic, has_latin

REQUEST_ID_PATTERN = r"^[A-Za-z0-9._-]{8,64}$"  # safe in logs and headers


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"question": "ما هي مدة تقادم الالتزام؟"},
                {"question": "Is a contract binding on the parties?"},
            ]
        },
    )

    question: str = Field(min_length=3, max_length=2000, description="Arabic or English question")
    book: str | None = Field(
        default=None,
        max_length=200,
        description="optional filter, e.g. 'الكتاب الثاني: العقود المسماة'",
    )

    @field_validator("question")
    @classmethod
    def _has_words(cls, value: str) -> str:
        value = value.strip()
        if not (has_arabic(value) or has_latin(value)):
            raise ValueError("question must contain Arabic or English words")
        return value


class Source(BaseModel):
    article_number: int
    citation: str
    citation_ar: str
    is_repealed: bool = False
    quality_flags: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    language: Literal["ar", "en"]
    sources: list[Source]
    refused: bool
    invalid_citations: list[int]
    request_id: str
    prompt_version: str
    model: str
    usage: dict[str, int]
    timings_ms: dict[str, float]
    guardrails: list[str] = Field(
        default_factory=list, description="guards that fired, e.g. pii:phone, injection:override"
    )


class ErrorBody(BaseModel):
    detail: str
    request_id: str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)  # same shape the API hands out
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    documents_indexed: int
    points: int
    index_collection: str
