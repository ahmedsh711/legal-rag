"""End-to-end checks on the real PDF. Skipped when the PDF is not present (e.g. CI without `dvc pull`)."""

import random
from pathlib import Path

import pytest
import yaml

from legalrag.ingest.parse import parse_pdf
from legalrag.ingest.validate import validate_articles

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "data" / "raw" / "egyptian_civil_code.pdf"
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="corpus PDF not present (run `dvc pull`)")


@pytest.fixture(scope="module")
def articles():
    return {a.article_number: a for a in parse_pdf(PDF)}


def test_every_article_number_is_present(articles):
    assert sorted(articles) == list(range(1, 1150))


def test_repealed_ranges_come_from_the_pdf_notes(articles):
    repealed = {n for n, a in articles.items() if a.is_repealed}
    assert repealed == set(range(54, 81)) | set(range(389, 418))


@pytest.mark.parametrize(
    ("number", "arabic", "english"),
    [
        (1, "تسرى النصوص التشريعية", "Provisions of laws govern"),
        (147, "العقد شريعة المتعاقدين", "The contract makes the law of the parties"),
        (374, "خمس عشرة سنة", "fifteen years"),
        (428, "يكف عن أي عمل", "The vendor is bound"),  # Arabic and English glued on one line
        (452, "تسقط بالتقادم دعوى الضمان", "An action on a warranty"),  # header was "rticle 452"
        (1149, "للشركاء الذين اقتسموا عقاراً", "Co-owners who have partitioned"),
    ],
)
def test_known_articles_have_the_right_text(articles, number, arabic, english):
    assert arabic in articles[number].text_ar
    assert english in articles[number].text_en


def test_hierarchy_is_captured(articles):
    a = articles[147]
    assert a.book.startswith("الكتاب الأول")
    assert a.chapter.startswith("الباب الأول")
    assert a.section.startswith("الفصل الأول")
    assert "BOOK I" in a.heading_en


def test_corpus_passes_validation(articles):
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))["corpus"]
    report = validate_articles(
        list(articles.values()),
        params["expected_articles"],
        params["max_chars"],
        {int(k): v for k, v in params["known_anomalies"].items()},
    )
    assert report.ok, report.errors[:5]


def test_random_spot_check_of_20_articles(articles):
    live = [a for a in articles.values() if not a.is_repealed and a.text_ar]
    for a in random.Random(42).sample(live, 20):
        assert "Article" not in a.text_ar and "مادة" not in a.text_ar[:6]
        assert a.text_en[:1].isupper() or a.text_en[:1] in "(*", a.text_en[:40]
        assert a.citation == f"Egyptian Civil Code, Article {a.article_number}"
