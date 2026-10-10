"""The drift job: compare a window of prediction events with the reference, decide, report."""

from __future__ import annotations

import random

from legalrag.monitoring.drift_job import compare, textfile_lines
from legalrag.monitoring.events import PredictionEvent

BOOKS = ["Book 1", "Book 2", "Book 3", "Book 4"]


def events(n: int, seed: int, ar_share: float = 0.5, books: list[str] = BOOKS,
           chars: tuple[int, int] = (20, 80), refused_share: float = 0.1) -> list[PredictionEvent]:  # fmt: skip
    rng = random.Random(seed)
    out = []
    for i in range(n):
        lang = "ar" if rng.random() < ar_share else "en"
        length = rng.randint(*chars)
        out.append(PredictionEvent(
            request_id=f"r{seed}-{i}", endpoint="ask", lang=lang, question_chars=length,
            question_words=max(1, length // 6), pii_redacted=False, top_articles=[1],
            top_book=rng.choice(books), answerable_score=None,
            refused=rng.random() < refused_share, guardrails=[], timings_ms={},
            prompt_version="v3", llm_model="m", index_collection="c", decider="none"))  # fmt: skip
    return out


def test_traffic_like_the_reference_is_not_drift():
    report = compare(events(300, seed=1), events(300, seed=2))
    assert not report.drift, [r for r in report.rows if r.drift]
    assert report.n_reference == 300 and report.n_current == 300


def test_a_topic_and_language_shift_is_drift_and_names_the_features():
    shifted = events(300, seed=3, ar_share=0.9, books=["Book 4"], chars=(60, 160),
                     refused_share=0.6)  # fmt: skip
    report = compare(events(300, seed=1), shifted)
    flagged = {r.feature for r in report.rows if r.drift}
    assert report.drift and {"lang", "top_book", "question_chars", "refused"} <= flagged


def test_too_few_current_events_are_not_judged():
    report = compare(events(300, seed=1), events(10, seed=2, ar_share=1.0), min_samples=50)
    assert not report.drift and report.reason == "too few samples (10 < 50)"


def test_textfile_has_one_score_per_test_and_the_alert_flag():
    report = compare(events(300, seed=1), events(300, seed=2))
    lines = textfile_lines(report, alert=False, last_trigger=None)
    assert 'rag_drift_score{feature="lang",test="chi2",kind="input"}' in "\n".join(lines)
    assert "rag_drift_alert 0" in lines and any(ln.startswith("# HELP") for ln in lines)
    assert not any(ln.startswith("rag_drift_last_trigger") for ln in lines)  # never triggered


def test_the_last_trigger_time_survives_later_quiet_runs():
    # found live: a second run 80 s later reset "this run triggered" to 0 before the alert's
    # for: 1m elapsed; the alert now reads the time of the last trigger instead
    from datetime import UTC, datetime

    report = compare(events(300, seed=1), events(300, seed=2))
    when = datetime(2026, 10, 10, 13, 9, 52, tzinfo=UTC)
    lines = textfile_lines(report, alert=False, last_trigger=when)
    assert f"rag_drift_last_trigger_timestamp_seconds {when.timestamp():.0f}" in lines


def test_input_drift_and_behaviour_drift_are_reported_apart():
    # same questions, but a new model refuses most of them: the inputs did not move, the outputs did
    report = compare(events(300, seed=1), events(300, seed=2, refused_share=0.9))
    inputs = [r for r in report.rows if r.kind == "input"]
    outputs = [r for r in report.rows if r.kind == "behaviour"]
    assert not any(r.drift for r in inputs)
    assert any(r.drift for r in outputs) and report.drift
    assert "behaviour" in report.reason
