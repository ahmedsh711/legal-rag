"""Qdrant layout for the articles index.

- One collection per index build, named ``articles_<hash>``; the alias ``articles`` points at the
  live one. Re-indexing builds a new collection and moves the alias in one atomic step, so a bad
  build never touches what is serving, and a rollback is moving the alias back.
- Each article becomes up to two points (Arabic text, English text) sharing one payload; a
  repealed article becomes one "note" point. Point ids are deterministic, so re-running upserts
  overwrites instead of duplicating.
- Index metadata (embedding model, normalization, source hash, git SHA) lives in a tiny side
  collection ``index_meta``, one point per collection. The API reads it at startup and refuses
  to start if its own embedding model does not match (the instructor's 18%-zero-hit incident).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from legalrag.index.embedder import Embedding
from legalrag.ingest.schema import Article

META_COLLECTION = "index_meta"
DENSE, SPARSE = "dense", "sparse"
# Bump when article_texts() or the payload changes: it is part of the collection name, so new
# text gives a new collection instead of silently reusing vectors of the old text.
TEXT_FORMAT_VERSION = "v1"
LANG_OFFSET = {"ar": 1, "en": 2, "note": 3}  # point id = article_number * 10 + offset

PAYLOAD_FIELDS = (
    "article_number", "book", "chapter", "section", "subsection", "topic", "heading_en",
    "text_ar", "text_en", "is_repealed", "note", "quality_flags", "source_page", "citation", "citation_ar",
)  # fmt: skip


def point_id(article_number: int, lang: str) -> int:
    return article_number * 10 + LANG_OFFSET[lang]


def article_texts(a: Article) -> list[tuple[str, str]]:
    """(lang, text to embed) for one article. Headings are prepended as context for retrieval."""
    if a.is_repealed:
        return [
            (
                "note",
                f"Article {a.article_number} (المادة {a.article_number}) is repealed. {a.note}",
            )
        ]
    out = []
    if a.text_ar:
        header = " / ".join(x for x in (a.section, a.topic) if x)
        out.append(
            (
                "ar",
                f"المادة {a.article_number}"
                + (f" - {header}" if header else "")
                + f"\n{a.text_ar}",
            )
        )
    if a.text_en:
        out.append(("en", f"Article {a.article_number} - {a.heading_en}\n{a.text_en}"))
    return out


def payload(a: Article, lang: str) -> dict[str, Any]:
    data = a.model_dump(include=set(PAYLOAD_FIELDS))
    data["lang"] = lang
    return data


def create_collection(client: QdrantClient, name: str, dim: int) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)  # rebuilds of the same hash start clean
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams()},
    )
    for field_name, schema in (
        ("article_number", models.PayloadSchemaType.INTEGER),
        ("is_repealed", models.PayloadSchemaType.BOOL),
        ("book", models.PayloadSchemaType.KEYWORD),
        ("lang", models.PayloadSchemaType.KEYWORD),
    ):
        client.create_payload_index(name, field_name, schema)


def upsert_points(
    client: QdrantClient,
    name: str,
    items: Sequence[tuple[Article, str]],
    embeddings: Sequence[Embedding],
    batch_size: int = 128,
) -> int:
    points = []
    for (article, lang), emb in zip(items, embeddings, strict=True):
        indices, values = emb.sparse_indices_values()
        points.append(
            models.PointStruct(
                id=point_id(article.article_number, lang),
                vector={
                    DENSE: emb.dense,
                    SPARSE: models.SparseVector(indices=indices, values=values),
                },
                payload=payload(article, lang),
            )
        )
    for start in range(0, len(points), batch_size):
        client.upsert(name, points[start : start + batch_size], wait=True)
    return len(points)


def swap_alias(client: QdrantClient, alias: str, collection: str) -> str | None:
    """Point ``alias`` at ``collection`` atomically; return the collection it pointed at before."""
    previous = alias_target(client, alias)
    ops: list[Any] = []
    if previous:
        ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    ops.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=ops)
    return previous


def alias_target(client: QdrantClient, alias: str) -> str | None:
    for a in client.get_aliases().aliases:
        if a.alias_name == alias:
            return a.collection_name
    return None


def _meta_id(collection: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"legal-rag/{collection}"))


def write_metadata(client: QdrantClient, collection: str, metadata: dict[str, Any]) -> None:
    if not client.collection_exists(META_COLLECTION):
        client.create_collection(META_COLLECTION, vectors_config={})
    client.upsert(
        META_COLLECTION,
        [
            models.PointStruct(
                id=_meta_id(collection), vector={}, payload={**metadata, "collection": collection}
            )
        ],
        wait=True,
    )


def read_metadata(client: QdrantClient, collection: str) -> dict[str, Any] | None:
    if not client.collection_exists(META_COLLECTION):
        return None
    found = client.retrieve(META_COLLECTION, [_meta_id(collection)], with_payload=True)
    return dict(found[0].payload) if found else None
