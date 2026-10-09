"""Corpus validation: a silent parsing bug must fail loudly here, not become a hallucination later."""

import json

import pytest

from legalrag.ingest.schema import Article
from legalrag.ingest.validate import ValidationReport, main, validate_articles


def make(n: int, **kw) -> Article:
    base = {
        "article_number": n,
        "text_ar": f"نص المادة {n}",
        "text_en": f"Text of article {n}.",
        "source_page": 1,
    }
    return Article(**{**base, **kw})


def corpus(total: int = 6) -> list[Article]:
    arts = [make(n) for n in range(1, total + 1)]
    arts[3] = make(4, text_ar="", text_en="", is_repealed=True, note="Articles 4-4 repealed")
    return arts


def test_clean_corpus_passes():
    report = validate_articles(corpus(), expected_total=6, max_chars=500)
    assert report.ok, report.errors
    assert report.metrics["articles_total"] == 6
    assert report.metrics["articles_repealed"] == 1
    assert report.metrics["articles_live"] == 5


def test_gap_is_an_error():
    arts = [a for a in corpus() if a.article_number != 5]
    report = validate_articles(arts, expected_total=6, max_chars=500)
    assert not report.ok
    assert any("missing" in e for e in report.errors)


def test_duplicate_is_an_error():
    report = validate_articles([*corpus(), make(2)], expected_total=6, max_chars=500)
    assert any("duplicate" in e for e in report.errors)


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("text_ar", "", "empty Arabic"),
        ("text_en", "", "empty English"),
        ("text_en", "English with عربي inside", "Arabic letters in English"),
        ("text_ar", "نص فيه some English words here", "Latin words in Arabic"),
        ("text_ar", "ن" * 600, "too long"),
    ],
)
def test_bad_live_article_is_an_error(field, value, fragment):
    arts = corpus()
    arts[1] = make(2, **{field: value})
    report = validate_articles(arts, expected_total=6, max_chars=500)
    assert not report.ok
    assert any(fragment in e and "2" in e for e in report.errors), report.errors


def test_known_anomaly_is_allowed_but_reported():
    arts = corpus()
    arts[1] = make(2, text_ar="")
    report = validate_articles(
        arts, expected_total=6, max_chars=500, known_anomalies={2: "merged in source"}
    )
    assert report.ok
    assert report.metrics["known_anomalies"] == 1
    assert any("2" in w for w in report.warnings)


def test_repealed_article_must_not_carry_text():
    arts = corpus()
    arts[3] = make(4, is_repealed=True, text_ar="نص", note="repealed")
    report = validate_articles(arts, expected_total=6, max_chars=500)
    assert not report.ok


def test_cli_writes_metrics_and_fails_on_errors(tmp_path):
    good = tmp_path / "articles.json"
    good.write_text(
        json.dumps([a.model_dump() for a in corpus()], ensure_ascii=False), encoding="utf-8"
    )
    params = tmp_path / "params.yaml"
    params.write_text(
        "corpus:\n  expected_articles: 6\n  max_chars: 500\n  known_anomalies: {}\n",
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.json"
    main(["--articles", str(good), "--params", str(params), "--metrics", str(metrics)])
    assert json.loads(metrics.read_text(encoding="utf-8"))["articles_total"] == 6

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps([a.model_dump() for a in corpus()[:-1]], ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc:
        main(["--articles", str(bad), "--params", str(params), "--metrics", str(metrics)])
    assert exc.value.code == 1


def test_report_type():
    assert isinstance(
        validate_articles(corpus(), expected_total=6, max_chars=500), ValidationReport
    )
