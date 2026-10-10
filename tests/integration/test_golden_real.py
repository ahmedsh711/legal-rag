"""Golden set against the real corpus; skipped without the DVC-tracked articles.json."""

from pathlib import Path

import pytest

from legalrag.eval.golden import check_golden, load_golden
from legalrag.ingest.validate import load_articles

ROOT = Path(__file__).resolve().parents[2]
ARTICLES = ROOT / "data" / "processed" / "articles.json"
pytestmark = pytest.mark.skipif(
    not ARTICLES.exists(), reason="articles.json not present (dvc repro)"
)


def test_every_gold_article_exists_with_the_right_repeal_status():
    corpus = {a.article_number: a for a in load_articles(ARTICLES)}
    items = load_golden(ROOT / "data" / "golden" / "golden_set.jsonl")
    assert check_golden(items, corpus) == []
