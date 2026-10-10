"""The drift job: compare a window of prediction events with the reference, decide, report."""

from __future__ import annotations

import random
from datetime import UTC, datetime

import pytest

from legalrag.monitoring.drift_job import aa_check, compare, parse_since, textfile_lines
from legalrag.monitoring.events import PredictionEvent

BOOKS = ["Book 1", "Book 2", "Book 3", "Book 4"]


def events(
    n: int,
    seed: int,
    ar_share: float = 0.5,
    books: list[str] = BOOKS,
    chars: tuple[int, int] = (20, 80),
    refused_share: float = 0.1,
    llm_model: str = "m",
    prompt_version: str = "v3",
) -> list[PredictionEvent]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        lang = "ar" if rng.random() < ar_share else "en"
        length = rng.randint(*chars)
        out.append(
            PredictionEvent(
                request_id=f"r{seed}-{i}",
                endpoint="ask",
                lang=lang,
                question_chars=length,
                question_words=max(1, length // 6),
                pii_redacted=False,
                top_articles=[1],
                top_book=rng.choice(books),
                answerable_score=None,
                refused=rng.random() < refused_share,
                guardrails=[],
                timings_ms={},
                prompt_version=prompt_version,
                llm_model=llm_model,
                index_collection="c",
                decider="none",
            )
        )
    return out


def flagged(report) -> set[str]:
    return {r.feature for r in report.rows if r.drift}


def test_traffic_like_the_reference_is_not_drift():
    report = compare(events(300, seed=1), events(300, seed=2))
    assert not report.drift, [r for r in report.rows if r.drift]
    assert report.n_reference == 300 and report.n_current == 300


def test_a_topic_and_language_shift_is_drift_and_names_the_features():
    shifted = events(
        300, seed=3, ar_share=0.9, books=["Book 4"], chars=(60, 160), refused_share=0.6
    )
    report = compare(events(300, seed=1), shifted)
    assert report.drift and {"lang", "top_book", "question_chars", "refused"} <= flagged(report)


def test_the_three_kinds_are_reported_apart():
    # same questions, but the system refuses most of them: inputs and retrieval did not move
    report = compare(events(300, seed=1), events(300, seed=2, refused_share=0.9))
    kinds = {r.feature: r.kind for r in report.rows}
    assert kinds["lang"] == "input" and kinds["top_book"] == "retrieval"
    assert kinds["refused"] == "behaviour"
    assert flagged(report) == {"refused"} and "behaviour drift" in report.reason


def test_a_model_change_is_not_called_behaviour_drift():
    # a vLLM window vs a Gemini reference is a model change, not drift
    current = events(300, seed=2, refused_share=0.9, llm_model="vllm-qwen")
    report = compare(events(300, seed=1), current)
    assert not any(r.kind == "behaviour" for r in report.rows)
    assert any("model or prompt changed" in note for note in report.notes)
    assert not report.drift


def test_too_few_current_events_are_not_judged():
    report = compare(events(300, seed=1), events(10, seed=2, ar_share=1.0), min_samples=50)
    assert not report.drift and report.reason == "too few samples (10 < 50)"


def test_an_empty_reference_fails_loudly():
    with pytest.raises(ValueError, match="reference"):
        compare([], events(100, seed=1))


def test_textfile_has_scores_the_latest_verdict_and_the_trigger_time():
    report = compare(events(300, seed=1), events(300, seed=2))
    lines = textfile_lines(report, alert=False, last_trigger=None)
    text = "\n".join(lines)
    assert 'rag_drift_score{feature="lang",test="chi2",kind="input"}' in text
    assert "rag_drift_detected 0" in lines and "rag_drift_alert 0" in lines
    assert not any(ln.startswith("rag_drift_last_trigger") for ln in lines)
    when = datetime(2026, 10, 10, 13, 9, 52, tzinfo=UTC)
    lines = textfile_lines(report, alert=False, last_trigger=when)
    assert f"rag_drift_last_trigger_timestamp_seconds {when.timestamp():.0f}" in lines


def test_aa_check_measures_false_alarms_on_two_halves_of_the_same_traffic():
    rates = aa_check(events(400, seed=5), runs=20, seed=0)
    assert set(rates) >= {"lang/chi2", "question_chars/ks", "inputs/mmd", "any"}
    assert all(0 <= v <= 1 for v in rates.values())
    assert rates["any"] <= 0.2  # one window split in two should rarely look like drift


def test_an_effect_size_alone_is_not_drift():
    # small windows make big-looking effects from noise, so an effect counts only when its
    # feature's test is significant too
    for seed in range(30):
        cur = events(25, seed=100 + seed)
        report = compare(events(25, seed=seed), cur, min_samples=1, seed=seed)
        significant = {r.feature for r in report.rows if r.drift and r.p_value is not None}
        effects = {r.feature for r in report.rows if r.drift and r.p_value is None}
        assert effects <= significant, (seed, report.rows)


def test_since_without_a_timezone_means_utc():
    assert parse_since("2026-10-10T13:10:00") == datetime(2026, 10, 10, 13, 10, tzinfo=UTC)
