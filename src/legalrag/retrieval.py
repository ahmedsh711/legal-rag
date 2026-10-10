"""Article retrieval: explicit article references plus hybrid dense + sparse search in Qdrant."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from qdrant_client import QdrantClient, models

from legalrag.index.embedder import Embedding
from legalrag.index.store import DENSE, SPARSE
from legalrag.ingest.normalize import arabic_digits_to_int

# "المادة 147", "المادة ١٤٧", "مادة 60", "Article 374", "art. 5"
# (?<![A-Za-z]) so "particle 5" is not "article 5"; (?![0-9٠-٩]) so "12345" is not "1234"
_ARTICLE_REF = re.compile(
    r"(?:الماد[ةه]|ماد[ةه]|(?<![A-Za-z])(?:article|art\.))\s*\(?\s*([0-9٠-٩]{1,4})(?![0-9٠-٩])",
    re.IGNORECASE,
)
PREFETCH_MULTIPLIER = 4  # each article has AR and EN points, so over-fetch before deduplicating


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str], batch_size: int = 16) -> list[Embedding]: ...


@dataclass(frozen=True)
class Chunk:
    """One article as seen by the generator and returned to the user as a source."""

    article_number: int
    citation: str
    citation_ar: str
    text_ar: str
    text_en: str
    is_repealed: bool = False
    note: str = ""
    book: str = ""
    chapter: str = ""
    section: str = ""
    topic: str = ""
    heading_en: str = ""
    source_page: int = 0
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    score: float = 0.0

    @classmethod
    def from_payload(cls, payload: dict[str, Any], score: float = 0.0) -> Chunk:
        keys = cls.__dataclass_fields__.keys() - {"score", "quality_flags"}
        return cls(
            **{k: payload.get(k, "") for k in keys if k in payload},
            quality_flags=tuple(payload.get("quality_flags", ())),
            score=score,
        )


def article_numbers_in(question: str, max_article: int = 1149) -> list[int]:
    """Article numbers the user mentions explicitly, in order, without duplicates."""
    numbers: list[int] = []
    for m in _ARTICLE_REF.finditer(question):
        n = arabic_digits_to_int(m.group(1))
        if 1 <= n <= max_article and n not in numbers:
            numbers.append(n)
    return numbers


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        alias: str,
        embedder: Embedder,
        top_n: int = 12,
        max_article: int = 1149,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",  # dense/sparse for ablations
    ):
        self.client, self.alias, self.embedder = client, alias, embedder
        self.top_n, self.max_article, self.mode = top_n, max_article, mode

    def _filter(self, book: str | None) -> models.Filter:
        must: list[Any] = [
            models.FieldCondition(key="is_repealed", match=models.MatchValue(value=False))
        ]
        if book:
            must.append(models.FieldCondition(key="book", match=models.MatchValue(value=book)))
        return models.Filter(must=must)

    def search(
        self, question: str, top_n: int | None = None, book: str | None = None
    ) -> list[Chunk]:
        """Hybrid dense + sparse search fused by RRF; one chunk per article, best first."""
        top_n = top_n or self.top_n
        emb = self.embedder.encode([question])[0]
        indices, values = emb.sparse_indices_values()
        flt = self._filter(book)
        limit = top_n * PREFETCH_MULTIPLIER
        queries = {
            DENSE: emb.dense,
            SPARSE: models.SparseVector(indices=indices, values=values),
        }
        if self.mode == "hybrid":
            result = self.client.query_points(
                self.alias,
                prefetch=[
                    models.Prefetch(query=q, using=name, limit=limit, filter=flt)
                    for name, q in queries.items()
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            name = DENSE if self.mode == "dense" else SPARSE
            result = self.client.query_points(
                self.alias,
                query=queries[name],
                using=name,
                query_filter=flt,
                limit=limit,
                with_payload=True,
            )
        chunks: dict[int, Chunk] = {}
        for point in result.points:  # already sorted by fused score
            n = point.payload["article_number"]
            if n not in chunks:
                chunks[n] = Chunk.from_payload(point.payload, point.score)
            if len(chunks) == top_n:
                break
        return list(chunks.values())

    def by_numbers(self, numbers: list[int]) -> list[Chunk]:
        """Fetch articles by number (repealed included), in the order asked."""
        if not numbers:
            return []
        points, _ = self.client.scroll(
            self.alias,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="article_number", match=models.MatchAny(any=numbers))
                ]
            ),
            limit=len(numbers) * 3,
            with_payload=True,
        )
        found = {p.payload["article_number"]: Chunk.from_payload(p.payload, 1.0) for p in points}
        return [found[n] for n in numbers if n in found]

    def retrieve(
        self, question: str, top_n: int | None = None, book: str | None = None
    ) -> list[Chunk]:
        """Explicitly referenced articles first, then hybrid search results, without duplicates."""
        top_n = top_n or self.top_n
        chunks = self.by_numbers(article_numbers_in(question, self.max_article))
        limit = max(top_n, len(chunks))  # explicit references are always kept
        seen = {c.article_number for c in chunks}
        for c in self.search(question, top_n=top_n, book=book):
            if len(chunks) >= limit:
                break
            if c.article_number not in seen:
                chunks.append(c)
                seen.add(c.article_number)
        return chunks
