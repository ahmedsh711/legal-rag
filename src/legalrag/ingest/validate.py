"""Validate articles.json before anything is embedded.

"A silent parsing bug becomes a hallucination three steps later, and by then you will blame
the model" (handbook). Rules, all from the corpus inspection:

- numbers 1..expected_total, each exactly once (repealed articles are records, not gaps);
- live articles have Arabic and English text, unless listed as a known source anomaly;
- no Arabic letters in English text, no Latin words in Arabic text (a failed column split);
- no record longer than ``max_chars`` per language (a failed article split);
- repealed articles carry a note and no text.

Run as a DVC stage; it writes metrics for ``dvc metrics show`` and exits 1 on any error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from legalrag.ingest.schema import Article
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)

_ARABIC_LETTER = re.compile(r"[ء-ي]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _check_live(a: Article, max_chars: int, anomaly: str | None, report: ValidationReport) -> None:
    n = a.article_number
    problems = []
    if not a.text_ar:
        problems.append(f"article {n}: empty Arabic text")
    if not a.text_en:
        problems.append(f"article {n}: empty English text")
    if _ARABIC_LETTER.search(a.text_en):
        problems.append(f"article {n}: Arabic letters in English text")
    if _LATIN_WORD.search(a.text_ar):
        problems.append(f"article {n}: Latin words in Arabic text")
    if max(len(a.text_ar), len(a.text_en)) > max_chars:
        problems.append(
            f"article {n}: text too long (> {max_chars} chars), probably a failed split"
        )
    if anomaly:
        report.warnings.extend(f"{p} (known anomaly: {anomaly})" for p in problems)
    else:
        report.errors.extend(problems)


def validate_articles(
    articles: list[Article],
    expected_total: int,
    max_chars: int,
    known_anomalies: dict[int, str] | None = None,
) -> ValidationReport:
    known_anomalies = known_anomalies or {}
    report = ValidationReport()
    numbers = [a.article_number for a in articles]
    seen: set[int] = set()
    for n in numbers:
        if n in seen:
            report.errors.append(f"duplicate article {n}")
        seen.add(n)
    missing = [n for n in range(1, expected_total + 1) if n not in seen]
    if missing:
        report.errors.append(
            f"missing articles: {missing[:20]}{' ...' if len(missing) > 20 else ''}"
        )
    extra = sorted(n for n in seen if n > expected_total)
    if extra:
        report.errors.append(f"unexpected article numbers above {expected_total}: {extra[:20]}")

    for a in articles:
        if a.is_repealed:
            if a.text_ar or not a.note:
                report.errors.append(
                    f"article {a.article_number}: repealed but has Arabic text or no note"
                )
        else:
            _check_live(a, max_chars, known_anomalies.get(a.article_number), report)

    live = [a for a in articles if not a.is_repealed]
    report.metrics = {
        "articles_total": len(articles),
        "articles_live": len(live),
        "articles_repealed": len(articles) - len(live),
        "known_anomalies": sum(1 for a in live if a.article_number in known_anomalies),
        "max_chars_ar": max((len(a.text_ar) for a in live), default=0),
        "max_chars_en": max((len(a.text_en) for a in live), default=0),
        "mean_chars_ar": round(sum(len(a.text_ar) for a in live) / max(len(live), 1), 1),
        "errors": len(report.errors),
        "warnings": len(report.warnings),
    }
    return report


def main(argv: list[str] | None = None) -> None:
    from legalrag.settings import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Validate articles.json")
    parser.add_argument("--articles", default=settings.articles_path)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--metrics", default="reports/corpus_metrics.json")
    args = parser.parse_args(argv)
    configure_logging(settings.log_level)

    params = yaml.safe_load(Path(args.params).read_text(encoding="utf-8"))["corpus"]
    raw = json.loads(Path(args.articles).read_text(encoding="utf-8"))
    articles = [Article(**r) for r in raw]
    anomalies = {int(k): v for k, v in (params.get("known_anomalies") or {}).items()}
    report = validate_articles(
        articles, params["expected_articles"], params["max_chars"], anomalies
    )

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report.metrics, indent=2), encoding="utf-8")
    for w in report.warnings:
        log.warning("corpus_warning", detail=w)
    for e in report.errors:
        log.error("corpus_error", detail=e)
    log.info("validated", ok=report.ok, **report.metrics)
    if not report.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
