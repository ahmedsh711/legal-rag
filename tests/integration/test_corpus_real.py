"""End-to-end checks on the real PDF. Skipped when the PDF is not present (e.g. CI without `dvc pull`)."""

import random
from pathlib import Path

import pytest

from legalrag.ingest.params import load_params
from legalrag.ingest.parse import apply_quality_flags, parse_pdf
from legalrag.ingest.validate import validate_articles

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "data" / "raw" / "egyptian_civil_code.pdf"
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="corpus PDF not present (run `dvc pull`)")


@pytest.fixture(scope="module")
def params():
    return load_params(ROOT / "params.yaml")


@pytest.fixture(scope="module")
def articles(params):
    return {a.article_number: a for a in apply_quality_flags(parse_pdf(PDF), params)}


def test_every_article_number_is_present(articles, params):
    assert sorted(articles) == list(range(1, params.expected_articles + 1))


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


def test_corpus_passes_validation(articles, params):
    report = validate_articles(list(articles.values()), params)
    assert report.ok, report.errors[:5]


def test_documented_anomalies_are_flagged(articles):
    assert articles[1022].quality_flags == ["arabic_text_inside_article_1021"]
    assert articles[1021].quality_flags == ["contains_arabic_of_article_1022"]
    assert articles[54].text_en == ""  # the repeal note is a note, not article text


def test_random_spot_check_of_20_articles(articles):
    live = [a for a in articles.values() if not a.is_repealed and a.text_ar]
    for a in random.Random(42).sample(live, 20):
        assert "Article" not in a.text_ar and "مادة" not in a.text_ar[:6]
        assert a.text_en, a.article_number
        assert a.text_en[0].isupper() or a.text_en[0] == "(", a.text_en[:40]
        assert a.citation == f"Egyptian Civil Code, Article {a.article_number}"
