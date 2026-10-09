"""Corpus validation: a silent parsing bug must fail loudly here, not become a hallucination later."""

import json

import pytest

from legalrag.ingest.params import Anomaly, CorpusInputError, CorpusParams, load_params
from legalrag.ingest.schema import Article
from legalrag.ingest.validate import load_articles, main, validate_articles

BOOK = "الكتاب الأول: الالتزامات"


def make(n: int, **kw) -> Article:
    base = {
        "article_number": n,
        "text_ar": f"نص المادة رقم {n} في القانون",
        "text_en": f"The text of article number {n} in the law.",
        "source_page": 1,
        "book": BOOK if n >= 3 else "",
    }
    return Article(**{**base, **kw})


def corpus(total: int = 6) -> list[Article]:
    arts = [make(n) for n in range(1, total + 1)]
    arts[3] = make(4, text_ar="", text_en="", is_repealed=True, note="Articles 4-4 repealed")
    return arts


def params(**kw) -> CorpusParams:
    base = {
        "expected_articles": 6,
        "max_chars_ar": 500,
        "max_chars_en": 500,
        "length_ratio": (0.5, 3.0),
        "pinned": {2: "رقم 2"},
        "book_starts": {3: "الكتاب الأول"},
    }
    return CorpusParams(**{**base, **kw})


def test_clean_corpus_passes():
    report = validate_articles(corpus(), params())
    assert report.ok, report.errors
    assert report.metrics["articles_total"] == 6
    assert report.metrics["articles_repealed"] == 1
    assert report.metrics["articles_live"] == 5


def test_gap_is_an_error():
    arts = [a for a in corpus() if a.article_number != 5]
    report = validate_articles(arts, params())
    assert any("missing" in e for e in report.errors)


def test_duplicate_is_an_error():
    report = validate_articles([*corpus(), make(2)], params())
    assert any("duplicate article 2" in e for e in report.errors)


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("text_ar", "", "empty Arabic"),
        ("text_ar", "   ", "empty Arabic"),  # whitespace-only counts as empty
        ("text_en", "", "empty English"),
        ("text_en", "English with عربي inside it", "Arabic letters in English"),
        ("text_ar", "نص فيه some English words رقم 2", "Latin words in Arabic"),
        ("text_ar", "ن" * 600, "too long"),
        ("text_en", "x" * 200, "length ratio"),  # English 200 chars vs short Arabic
    ],
)
def test_bad_live_article_is_an_error(field, value, fragment):
    arts = corpus()
    arts[1] = make(2, **{field: value})
    report = validate_articles(arts, params(pinned={}))
    assert not report.ok
    assert any(fragment in e and "article 2" in e for e in report.errors), report.errors


def test_anomaly_allows_only_the_listed_check():
    arts = corpus()
    arts[1] = make(2, text_ar="")
    anomaly = {2: Anomaly(reason="merged in source", flag="merged", allow=["empty_ar"])}
    report = validate_articles(arts, params(pinned={}, known_anomalies=anomaly))
    assert report.ok
    assert report.metrics["known_anomalies"] == 1
    assert any("article 2" in w for w in report.warnings)

    arts[1] = make(2, text_ar="", text_en="")  # a *new* defect on the same article still fails
    report = validate_articles(arts, params(pinned={}, known_anomalies=anomaly))
    assert any("empty English" in e for e in report.errors)


def test_repealed_article_must_not_carry_text():
    for kw in ({"text_ar": "نص"}, {"text_en": "text"}, {"note": ""}):
        arts = corpus()
        arts[3] = make(4, **{"is_repealed": True, "text_ar": "", "text_en": "", "note": "x", **kw})
        assert not validate_articles(arts, params()).ok


def test_pinned_phrase_and_book_start():
    arts = corpus()
    arts[1] = make(2, text_ar="نص آخر تمامًا بدون العبارة")
    report = validate_articles(arts, params())
    assert any("pinned phrase" in e for e in report.errors)

    arts = corpus()
    arts[1] = make(2, book=BOOK)  # book starts one article too early
    report = validate_articles(arts, params())
    assert any("should start here, not earlier" in e for e in report.errors)


def test_book_that_reappears_is_an_error():
    arts = corpus()
    arts[4] = make(5, book="الكتاب الثاني: العقود")
    arts[5] = make(6, book=BOOK)  # back to book 1 after book 2 started
    report = validate_articles(arts, params())
    assert any("reappears" in e for e in report.errors)


def test_load_articles_reports_bad_records(tmp_path):
    bad = tmp_path / "a.json"
    bad.write_text(
        json.dumps([{"article_number": 1, "source_page": 1, "surprise": 1}]), encoding="utf-8"
    )
    with pytest.raises(CorpusInputError, match="record #0"):
        load_articles(bad)
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CorpusInputError, match="not valid JSON"):
        load_articles(bad)
    with pytest.raises(CorpusInputError, match="not found"):
        load_articles(tmp_path / "missing.json")


def test_load_params_errors(tmp_path):
    p = tmp_path / "params.yaml"
    with pytest.raises(CorpusInputError, match="not found"):
        load_params(p)
    p.write_text("", encoding="utf-8")
    with pytest.raises(CorpusInputError, match="no 'corpus'"):
        load_params(p)
    p.write_text("corpus:\n  expected_articles: -1\n", encoding="utf-8")
    with pytest.raises(CorpusInputError, match="invalid corpus params"):
        load_params(p)


def write_params(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text(
        "corpus:\n  expected_articles: 6\n  max_chars_ar: 500\n  max_chars_en: 500\n"
        "  length_ratio: [0.5, 3.0]\n",
        encoding="utf-8",
    )
    return p


def test_cli_writes_metrics_and_fails_on_errors(tmp_path):
    p = write_params(tmp_path)
    good = tmp_path / "articles.json"
    good.write_text(
        json.dumps([a.model_dump() for a in corpus()], ensure_ascii=False), encoding="utf-8"
    )
    metrics = tmp_path / "metrics.json"
    main(["--articles", str(good), "--params", str(p), "--metrics", str(metrics)])
    assert json.loads(metrics.read_text(encoding="utf-8"))["articles_total"] == 6
    assert b"\r" not in metrics.read_bytes()

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps([a.model_dump() for a in corpus()[:-1]], ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc:
        main(["--articles", str(bad), "--params", str(p), "--metrics", str(metrics)])
    assert exc.value.code == 1


def test_cli_input_error_overwrites_stale_metrics(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"errors": 0}', encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "--articles",
                str(tmp_path / "nope.json"),
                "--params",
                str(write_params(tmp_path)),
                "--metrics",
                str(metrics),
            ]
        )
    assert json.loads(metrics.read_text(encoding="utf-8"))["errors"] == 1


def test_article_ids_and_citations_are_derived_and_checked():
    a = make(147)
    assert (a.id, a.citation) == ("eg-civil-147", "Egyptian Civil Code, Article 147")
    assert a.citation_ar == "القانون المدني المصري، المادة 147"
    with pytest.raises(ValueError, match="does not match"):
        Article(article_number=147, source_page=1, citation="Egyptian Civil Code, Article 148")
