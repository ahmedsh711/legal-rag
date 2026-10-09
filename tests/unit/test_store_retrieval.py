"""Index layout and hybrid retrieval on an in-memory Qdrant (no server, no model)."""

import pytest

from legalrag.index.store import (
    alias_target,
    article_texts,
    point_id,
    read_metadata,
    swap_alias,
    write_metadata,
)
from legalrag.retrieval import Retriever, article_numbers_in


def test_point_ids_are_deterministic_per_language():
    assert (
        point_id(147, "ar") == 1471 and point_id(147, "en") == 1472 and point_id(60, "note") == 603
    )


def test_repealed_article_becomes_one_note_point(articles):
    repealed = next(a for a in articles if a.is_repealed)
    texts = article_texts(repealed)
    assert [lang for lang, _ in texts] == ["note"]
    assert "repealed" in texts[0][1]


def test_live_article_text_carries_its_heading(articles):
    texts = dict(article_texts(articles[0]))
    assert texts["ar"].startswith("المادة 147 - الفصل الأول: العقد / آثار العقد")
    assert texts["en"].startswith("Article 147")


def test_metadata_round_trip_and_alias(qdrant):
    assert alias_target(qdrant, "articles") == "articles_test"
    assert read_metadata(qdrant, "articles_test")["embedding_model"] == "fake/bag-of-words"
    assert read_metadata(qdrant, "missing") is None


def test_swap_alias_returns_previous_collection(qdrant, embedder):
    from legalrag.index.store import create_collection

    create_collection(qdrant, "articles_v2", embedder.dim)
    write_metadata(qdrant, "articles_v2", {"embedding_model": "x"})
    assert swap_alias(qdrant, "articles", "articles_v2") == "articles_test"
    assert alias_target(qdrant, "articles") == "articles_v2"


@pytest.mark.parametrize(
    ("question", "numbers"),
    [
        ("ماذا تقول المادة ١٤٧ عن العقد؟", [147]),
        ("What does Article 374 say?", [374]),
        ("قارن المادة 418 والمادة 558", [418, 558]),
        ("ما مدة التقادم؟", []),
        ("Article 99999", []),  # out of range
        ("Article 12345", []),  # not "Article 1234"
        ("a particle 5 mm wide", []),  # "article" inside another word
        ("Art. 60 and art.61", [60, 61]),
    ],
)
def test_article_numbers_in_question(question, numbers):
    assert article_numbers_in(question, max_article=1149) == numbers


@pytest.fixture
def retriever(qdrant, embedder):
    return Retriever(qdrant, alias="articles", embedder=embedder, top_n=3)


def test_hybrid_search_finds_the_right_article(retriever):
    chunks = retriever.search("ما مدة التقادم خمس عشرة سنة")
    assert chunks[0].article_number == 374
    assert chunks[0].citation == "Egyptian Civil Code, Article 374"


def test_one_chunk_per_article_even_with_two_languages(retriever):
    chunks = retriever.search("contract law of the parties العقد شريعة المتعاقدين")
    numbers = [c.article_number for c in chunks]
    assert numbers.count(147) == 1
    assert chunks[0].text_ar and chunks[0].text_en


def test_repealed_articles_are_excluded_from_search_but_found_by_number(retriever):
    assert all(not c.is_repealed for c in retriever.search("repealed Presidential Decree"))
    by_number = retriever.by_numbers([60, 147])
    assert [c.article_number for c in by_number] == [60, 147]
    assert by_number[0].is_repealed and "repealed" in by_number[0].note


def test_book_filter(retriever):
    chunks = retriever.search("عقد", book="الكتاب الثاني")
    assert chunks and {c.article_number for c in chunks} <= {418, 558}


def test_retrieve_merges_explicit_article_references_first(retriever):
    chunks = retriever.retrieve("ماذا تقول المادة 558؟ وما مدة التقادم خمس عشرة سنة")
    assert chunks[0].article_number == 558
    assert 374 in [c.article_number for c in chunks]
