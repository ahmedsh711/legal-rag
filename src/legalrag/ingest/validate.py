"""Validate articles.json before anything is embedded.

"A silent parsing bug becomes a hallucination three steps later, and by then you will blame
the model" (handbook). Checks, each derived from what the corpus inspection found:

Per article (codes in ``CheckCode``; a documented anomaly may *allow* specific codes):
- ``empty_ar`` / ``empty_en``: live articles need both texts;
- ``arabic_in_en`` / ``latin_in_ar``: mixed scripts mean the column split failed;
- ``too_long``: longer than ``max_chars_*`` means an article split failed;
- ``length_ratio``: English/Arabic length outside the measured band means text moved
  between articles even though the counts look right.
Repealed articles must carry a note and no text.

Whole corpus:
- numbers 1..expected, each exactly once;
- pinned golden phrases are present in their articles;
- each book starts at its documented first article;
- a book or chapter never reappears after another one started (hierarchy is contiguous).

Run as a DVC stage; it writes metrics for ``dvc metrics show`` and exits 1 on any error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from legalrag.ingest.normalize import LATIN_WORD, has_arabic
from legalrag.ingest.params import CheckCode, CorpusInputError, CorpusParams, load_params
from legalrag.ingest.schema import Article
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------- per-article checks


def _article_problems(a: Article, params: CorpusParams) -> list[tuple[CheckCode, str]]:
    """Every rule a live article breaks, as (check code, message)."""
    n = a.article_number
    ar, en = a.text_ar.strip(), a.text_en.strip()
    problems: list[tuple[CheckCode, str]] = []
    if not ar:
        problems.append(("empty_ar", f"article {n}: empty Arabic text"))
    if not en:
        problems.append(("empty_en", f"article {n}: empty English text"))
    if has_arabic(en):
        problems.append(("arabic_in_en", f"article {n}: Arabic letters in English text"))
    if LATIN_WORD.search(ar):
        problems.append(("latin_in_ar", f"article {n}: Latin words in Arabic text"))
    if len(ar) > params.max_chars_ar or len(en) > params.max_chars_en:
        problems.append(("too_long", f"article {n}: text too long, probably a failed split"))
    if ar and en:
        ratio = len(en) / len(ar)
        low, high = params.length_ratio
        if not low <= ratio <= high:
            problems.append(
                (
                    "length_ratio",
                    f"article {n}: English/Arabic length ratio {ratio:.2f} outside {low}-{high}",
                )
            )
    return problems


def _check_article(a: Article, params: CorpusParams, report: ValidationReport) -> None:
    n = a.article_number
    if a.is_repealed:
        if a.text_ar.strip() or a.text_en.strip() or not a.note:
            report.errors.append(f"article {n}: repealed articles need a note and no text")
        return
    anomaly = params.known_anomalies.get(n)
    for code, message in _article_problems(a, params):
        if anomaly and code in anomaly.allow:
            report.warnings.append(f"{message} (known anomaly: {anomaly.reason})")
        else:
            report.errors.append(message)


# ---------------------------------------------------------------- corpus-level checks


def _check_numbers(articles: list[Article], expected: int, report: ValidationReport) -> None:
    counts = Counter(a.article_number for a in articles)
    report.errors.extend(f"duplicate article {n}" for n, c in sorted(counts.items()) if c > 1)
    missing = [n for n in range(1, expected + 1) if n not in counts]
    if missing:
        report.errors.append(
            f"missing articles: {missing[:20]}{' ...' if len(missing) > 20 else ''}"
        )
    extra = sorted(n for n in counts if n > expected)
    if extra:
        report.errors.append(f"unexpected article numbers above {expected}: {extra[:20]}")


def _check_pinned(
    by_number: dict[int, Article], params: CorpusParams, report: ValidationReport
) -> None:
    for n, phrase in params.pinned.items():
        if n not in by_number or phrase not in by_number[n].text_ar:
            report.errors.append(f"article {n}: pinned phrase {phrase!r} not found")
    for n, book_prefix in params.book_starts.items():
        here, before = by_number.get(n), by_number.get(n - 1)
        if here is None or not here.book.startswith(book_prefix):
            report.errors.append(f"article {n}: should start {book_prefix!r}")
        elif before is not None and before.book == here.book:
            report.errors.append(f"article {n}: {book_prefix!r} should start here, not earlier")


def _check_contiguous(articles: list[Article], report: ValidationReport) -> None:
    """A book or (book, chapter) that reappears after another started means headings were misread."""
    for name, key in (("book", lambda a: a.book), ("chapter", lambda a: (a.book, a.chapter))):
        seen, previous = set(), None
        for a in sorted(articles, key=lambda x: x.article_number):
            value = key(a)
            if value != previous:
                if value in seen:
                    report.errors.append(f"article {a.article_number}: {name} {value!r} reappears")
                seen.add(value)
                previous = value


def validate_articles(articles: list[Article], params: CorpusParams) -> ValidationReport:
    report = ValidationReport()
    _check_numbers(articles, params.expected_articles, report)
    by_number = {a.article_number: a for a in articles}
    for a in articles:
        _check_article(a, params, report)
    _check_pinned(by_number, params, report)
    _check_contiguous(articles, report)

    live = [a for a in articles if not a.is_repealed]
    report.metrics = {
        "articles_total": len(articles),
        "articles_live": len(live),
        "articles_repealed": len(articles) - len(live),
        "known_anomalies": sum(1 for a in live if a.article_number in params.known_anomalies),
        "max_chars_ar": max((len(a.text_ar) for a in live), default=0),
        "max_chars_en": max((len(a.text_en) for a in live), default=0),
        "mean_chars_ar": round(sum(len(a.text_ar) for a in live) / max(len(live), 1), 1),
        "errors": len(report.errors),
        "warnings": len(report.warnings),
    }
    return report


# ---------------------------------------------------------------- CLI (DVC stage)


def load_articles(path: str | Path) -> list[Article]:
    """Read articles.json; a bad record is reported with its position, not as a wall of errors."""
    path = Path(path)
    if not path.is_file():
        raise CorpusInputError(f"articles file not found: {path} (run `dvc repro` or `dvc pull`)")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusInputError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise CorpusInputError(f"{path} must contain a JSON list of articles")
    articles = []
    for i, record in enumerate(raw):
        try:
            articles.append(Article(**record))
        except (ValidationError, TypeError) as exc:
            raise CorpusInputError(
                f"record #{i} in {path} does not match the schema: {exc}"
            ) from exc
    return articles


def _write_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate articles.json")
    parser.add_argument("--articles", default=None, help="default: settings.articles_path")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--metrics", default="reports/corpus_metrics.json")
    args = parser.parse_args(argv)

    from legalrag.settings import get_settings  # after argparse so --help works without a valid env

    settings = get_settings()
    configure_logging(settings.log_level)
    metrics_path = Path(args.metrics)
    try:
        params = load_params(args.params)
        articles = load_articles(args.articles or settings.articles_path)
    except CorpusInputError as exc:
        log.error("corpus_input_error", detail=str(exc))
        _write_metrics(metrics_path, {"errors": 1})  # never leave a stale "all good" metrics file
        sys.exit(1)

    report = validate_articles(articles, params)
    _write_metrics(metrics_path, report.metrics)
    for w in report.warnings:
        log.warning("corpus_warning", detail=w)
    for e in report.errors:
        log.error("corpus_error", detail=e)
    log.info("validated", ok=report.ok, **report.metrics)
    if not report.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
