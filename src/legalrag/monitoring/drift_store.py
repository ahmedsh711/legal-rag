"""Drift history in Postgres: one row per run (``drift_runs``) and per test (``drift_metrics``).

Grafana queries these tables directly. They are created on first use.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS drift_runs (
    run_at       timestamptz PRIMARY KEY,
    window_hours double precision NOT NULL,
    n_reference  integer NOT NULL,
    n_current    integer NOT NULL,
    drift        boolean NOT NULL,
    triggered    boolean NOT NULL,
    reason       text NOT NULL,
    guard        text NOT NULL
);
CREATE TABLE IF NOT EXISTS drift_metrics (
    run_at    timestamptz NOT NULL REFERENCES drift_runs (run_at) ON DELETE CASCADE,
    feature   text NOT NULL,
    test      text NOT NULL,
    statistic double precision NOT NULL,
    p_value   double precision,
    drift     boolean NOT NULL,
    PRIMARY KEY (run_at, feature, test)
);
"""


def store_run(db_url: str, run: Mapping[str, Any]) -> None:
    import psycopg

    with psycopg.connect(db_url, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute(
            "INSERT INTO drift_runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                run["run_at"],
                run["window_hours"],
                run["n_reference"],
                run["n_current"],
                run["drift"],
                run["triggered"],
                run["reason"],
                run["guard"],
            ),
        )
        cur.executemany(
            "INSERT INTO drift_metrics VALUES (%s, %s, %s, %s, %s, %s)",
            [
                (run["run_at"], r["feature"], r["test"], r["statistic"], r["p_value"], r["drift"])
                for r in run["rows"]
            ],
        )
