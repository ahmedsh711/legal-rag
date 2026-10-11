"""Daily drift check; when drift passes the storm guard, start the retraining workflow on GitHub."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import httpx
from _common import DEFAULT_ARGS, HOST_DIR, app_task
from airflow.exceptions import AirflowSkipException
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import dag, task
from docker.types import Mount

REPO = os.environ.get("GITHUB_REPOSITORY", "ahmedsh711/legal-rag")
log = logging.getLogger(__name__)


@dag(
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["legal-rag", "monitoring"],
)
def monitoring():
    drift = app_task(
        "drift_job",
        [
            "legalrag.monitoring.drift_job",
            "run",
            "--hours",
            "24",
            "--events",
            "/app/events",
            "--reference",
            "/app/data/monitoring/reference_events.jsonl",
            "--out",
            "/app/reports/drift",
            "--state",
            "/app/reports/drift",
            "--textfile",
            "/textfile/drift.prom",
        ],
        extra_mounts=[
            Mount("/app/events", "legal-rag_events", type="volume", read_only=True),
            Mount("/textfile", f"{HOST_DIR}/monitoring/textfile", type="bind"),
        ],
        private_environment={"MONITORING_DB_URL": os.environ.get("MONITORING_DB_URL", "")},
        do_xcom_push=True,
    )

    @task.branch
    def route(last_log_line: str) -> str:
        run = json.loads(last_log_line)
        log.info("drift run: %s", json.dumps(run))
        return "dispatch_retraining" if run.get("triggered") else "no_action"

    @task
    def dispatch_retraining(last_log_line: str) -> None:
        token = os.environ.get("GITHUB_DISPATCH_TOKEN")
        if not token:
            raise AirflowSkipException("GITHUB_DISPATCH_TOKEN is not set")
        run = json.loads(last_log_line)
        response = httpx.post(
            f"https://api.github.com/repos/{REPO}/dispatches",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            json={"event_type": "drift-detected", "client_payload": {"reason": run.get("reason")}},
            timeout=30,
        )
        response.raise_for_status()

    line = drift.output
    route(line) >> [dispatch_retraining(line), EmptyOperator(task_id="no_action")]


monitoring()
