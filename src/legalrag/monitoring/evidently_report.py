"""Evidently data-drift report on the prediction-event features, next to the drift job's report.

Evidently picks its own test per column (K-S, Z-test, ...) at its default thresholds and with no
multiple-testing correction, so it is something to browse; the drift job's verdict is the alert.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from legalrag.monitoring.events import PredictionEvent
from legalrag.monitoring.files import write_atomic

NUMERICAL = ["question_chars", "query_cosine"]
CATEGORICAL = ["lang", "top_book", "refused"]


def _frame(events: Sequence[PredictionEvent]) -> Any:
    import pandas as pd

    columns = CATEGORICAL + NUMERICAL
    rows = [e.model_dump(include=set(columns)) for e in events]
    return pd.DataFrame(rows, columns=columns).astype({"query_cosine": float})  # None -> NaN


def summarize(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Per-column drift from ``Snapshot.dict()``: Evidently's method, threshold, score, verdict."""
    failed = {t["metric_config"]["metric_id"] for t in snapshot["tests"] if t["status"] == "FAIL"}
    summary: dict[str, Any] = {"dataset": None, "columns": {}}
    for metric in snapshot["metrics"]:
        config = metric["config"]
        if config["type"].endswith(":ValueDrift"):
            summary["columns"][config["column"]] = {
                "method": config["method"],
                "threshold": config["threshold"],
                "score": metric["value"],
                "drift": metric["id"] in failed,
            }
        elif config["type"].endswith(":DriftedColumnsCount"):
            summary["dataset"] = {**metric["value"], "drift": metric["id"] in failed}
    return summary


def write_report(
    reference: Sequence[PredictionEvent], current: Sequence[PredictionEvent], out: Path, stem: str
) -> dict[str, Any]:
    """Write ``<stem>.evidently.html`` and ``<stem>.evidently.json`` to ``out``; return the JSON."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    ref, cur = _frame(reference), _frame(current)
    # Evidently fails on a column with no values, e.g. query_cosine before the API recorded it
    skipped = [c for c in ref.columns if ref[c].isna().all() or cur[c].isna().all()]
    definition = DataDefinition(
        numerical_columns=[c for c in NUMERICAL if c not in skipped],
        categorical_columns=[c for c in CATEGORICAL if c not in skipped],
    )
    snapshot = Report([DataDriftPreset()], include_tests=True).run(
        current_data=Dataset.from_pandas(cur.drop(columns=skipped), data_definition=definition),
        reference_data=Dataset.from_pandas(ref.drop(columns=skipped), data_definition=definition),
    )
    summary = {**summarize(snapshot.dict()), "skipped": skipped}
    write_atomic(out / f"{stem}.evidently.html", snapshot.get_html_str(as_iframe=False))
    write_atomic(out / f"{stem}.evidently.json", json.dumps(summary, indent=2) + "\n")
    return summary
