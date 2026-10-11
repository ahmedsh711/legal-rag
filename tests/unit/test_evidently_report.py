"""Evidently's data-drift report on the event features, written next to the drift job's own."""

from __future__ import annotations

import json

from legalrag.monitoring.evidently_report import write_report
from tests.unit.test_drift_job import events


def test_the_html_report_and_a_per_column_summary_are_written(tmp_path):
    reference = events(200, seed=1, cosine=(0.6, 0.05))
    current = events(200, seed=2, ar_share=0.95, cosine=(0.6, 0.05))
    summary = write_report(reference, current, tmp_path, "20261011T120000Z")
    html = (tmp_path / "20261011T120000Z.evidently.html").read_text(encoding="utf-8")
    saved = (tmp_path / "20261011T120000Z.evidently.json").read_text(encoding="utf-8")
    assert "<html" in html.lower() and json.loads(saved) == summary
    columns = summary["columns"]
    assert set(columns) == {"lang", "question_chars", "top_book", "refused", "query_cosine"}
    assert columns["lang"]["drift"] and not columns["query_cosine"]["drift"]
    assert columns["lang"]["score"] < columns["lang"]["threshold"]  # a p-value at this size
    assert summary["dataset"]["count"] >= 1 and summary["skipped"] == []


def test_a_feature_missing_from_a_window_is_skipped_not_fatal(tmp_path):
    # events written before the API recorded query_cosine
    reference, current = events(100, seed=1, cosine=(0.6, 0.05)), events(100, seed=2)
    summary = write_report(reference, current, tmp_path, "run")
    assert "query_cosine" not in summary["columns"] and summary["skipped"] == ["query_cosine"]
