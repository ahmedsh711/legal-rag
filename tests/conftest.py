"""Shared fixtures: a tiny article set, a fake embedder and an in-memory Qdrant index."""

from __future__ import annotations

import hashlib
import math
import re

import pytest
from qdrant_client import QdrantClient

from legalrag.index.embedder import Embedding
from legalrag.index.store import (
    article_texts,
    create_collection,
    swap_alias,
    upsert_points,
    write_metadata,
)
from legalrag.ingest.schema import Article

DIM = 32


class FakeEmbedder:
    """Deterministic bag-of-words embedder: same words -> similar vectors. No model download."""

    model_name = "fake/bag-of-words"
    revision = "test"
    dim = DIM

    def encode(self, texts, batch_size: int = 16):
        out = []
        for text in texts:
            words = re.findall(r"\w+", text.lower())
            dense = [0.0] * DIM
            sparse: dict[int, float] = {}
            for w in words:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                dense[h % DIM] += 1.0
                sparse[h % 50_000] = sparse.get(h % 50_000, 0.0) + 1.0
            norm = math.sqrt(sum(x * x for x in dense)) or 1.0
            out.append(Embedding(dense=[x / norm for x in dense], sparse=sparse))
        return out


def art(n: int, ar: str, en: str, **kw) -> Article:
    return Article(article_number=n, text_ar=ar, text_en=en, source_page=1, **kw)


SAMPLE_ARTICLES = [
    art(
        147,
        "العقد شريعة المتعاقدين فلا يجوز نقضه ولا تعديله",
        "The contract makes the law of the parties",
        book="الكتاب الأول",
        section="الفصل الأول: العقد",
        topic="آثار العقد",
    ),
    art(
        374,
        "يتقادم الالتزام بانقضاء خمس عشرة سنة",
        "The term of prescription for obligations is fifteen years",
        book="الكتاب الأول",
        topic="التقادم المسقط",
    ),
    art(
        418,
        "البيع عقد يلتزم به البائع أن ينقل للمشتري ملكية شيء",
        "Sale is a contract whereby the vendor transfers ownership",
        book="الكتاب الثاني",
        topic="البيع",
    ),
    art(
        558,
        "الإيجار عقد يلتزم المؤجر بمقتضاه أن يمكن المستأجر من الانتفاع",
        "A lease is a contract by which the lessor undertakes",
        book="الكتاب الثاني",
        topic="الإيجار",
    ),
    Article(
        article_number=60,
        is_repealed=True,
        note="Articles 54-80 have been repealed by Presidential Decree.",
        source_page=7,
        book="",
    ),
]


@pytest.fixture
def articles() -> list[Article]:
    return list(SAMPLE_ARTICLES)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def qdrant(articles, embedder) -> QdrantClient:
    client = QdrantClient(":memory:")
    items = [(a, lang) for a in articles for lang, _ in article_texts(a)]
    texts = [text for a in articles for _, text in article_texts(a)]
    create_collection(client, "articles_test", embedder.dim)
    upsert_points(client, "articles_test", items, embedder.encode(texts))
    write_metadata(
        client, "articles_test", {"embedding_model": embedder.model_name, "normalization": "none"}
    )
    swap_alias(client, "articles", "articles_test")
    return client
