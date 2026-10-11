"""Retrieval on a real bge-m3 index built from the sample articles (the CI index job).

    uv run python -m legalrag.index.build --articles tests/fixtures/articles_sample.json
    INDEX_ALIAS=articles uv run pytest tests/integration/test_index_live.py --no-cov

The sample holds the gold article of every in-scope and repealed golden question. Skipped unless
INDEX_ALIAS names the alias the build moved.
"""

import os
from pathlib import Path

import pytest

from legalrag.eval.golden import load_golden
from legalrag.ingest.validate import load_articles

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "articles_sample.json"
ALIAS = os.environ.get("INDEX_ALIAS")
pytestmark = pytest.mark.skipif(not ALIAS, reason="needs an index of the sample: set INDEX_ALIAS")

IN_SAMPLE = {a.article_number for a in load_articles(SAMPLE)}
QUESTIONS = [
    it
    for it in load_golden(ROOT / "data" / "golden" / "golden_set.jsonl")
    if it.category == "in_scope" and set(it.gold_articles) <= IN_SAMPLE
]


@pytest.fixture(scope="module")
def retriever():
    from qdrant_client import QdrantClient

    from legalrag.index.embedder import BgeM3Embedder
    from legalrag.retrieval import Retriever
    from legalrag.settings import get_settings

    s = get_settings()
    embedder = BgeM3Embedder(
        s.embedding_model, s.embedding_device, s.query_max_length, revision=s.embedding_revision
    )
    return Retriever(QdrantClient(url=s.qdrant_url), ALIAS, embedder, mode=s.retrieval_mode)


def test_there_are_questions_to_ask():
    assert QUESTIONS  # an empty parametrization would pass silently


@pytest.mark.parametrize("item", QUESTIONS, ids=lambda it: it.id)
def test_the_gold_article_comes_first(retriever, item):
    assert retriever.retrieve(item.question)[0].article_number in item.gold_articles
