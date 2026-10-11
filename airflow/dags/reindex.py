"""Rebuild the index when a new Civil Code PDF lands in data/incoming/.

The new index is built behind a candidate alias and evaluated next to production on the golden
set (retrieval only, no LLM calls). Production moves to it only if no language got worse.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from _common import CANDIDATE_ALIAS, DEFAULT_ARGS, PRODUCTION_ALIAS, app_task
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.sdk import dag, task

INCOMING = Path("/opt/airflow/incoming")
PDF = "civil_code.pdf"
KEY = "{{ ti.xcom_pull(task_ids='run_key') }}"
WORK = f"/app/data/airflow/{KEY}"
log = logging.getLogger(__name__)
EVAL = ["legalrag.eval.run", "--retrieval-only", "--decider", "none", "--mode", "dense"]


@dag(
    schedule="@hourly",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["legal-rag", "index"],
)
def reindex():
    wait = FileSensor(
        task_id="wait_for_pdf",
        filepath=str(INCOMING / PDF),
        poke_interval=60,
        timeout=50 * 60,
        mode="reschedule",
        soft_fail=True,  # no new PDF this hour: skip the run instead of failing it
    )

    @task
    def run_key(run_id: str | None = None) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "-", run_id or "manual")

    parse = app_task(
        "parse",
        [
            "legalrag.ingest.parse",
            "--pdf",
            f"/app/data/incoming/{PDF}",
            "--out",
            f"{WORK}/articles.json",
        ],
    )
    validate = app_task(
        "validate",
        [
            "legalrag.ingest.validate",
            "--articles",
            f"{WORK}/articles.json",
            "--metrics",
            f"{WORK}/corpus.json",
        ],
        retries=0,  # a corpus that fails validation will fail again
    )
    build = app_task(
        "build_candidate",
        [
            "legalrag.index.build",
            "--articles",
            f"{WORK}/articles.json",
            "--manifest",
            f"{WORK}/manifest.json",
            "--metrics",
            f"{WORK}/index.json",
            "--alias",
            CANDIDATE_ALIAS,
        ],
    )
    eval_candidate = app_task(
        "eval_candidate", [*EVAL, "--name", f"{KEY}-candidate"], alias=CANDIDATE_ALIAS
    )
    eval_production = app_task("eval_production", [*EVAL, "--name", f"{KEY}-production"])
    compare = app_task(
        "compare",
        [
            "legalrag.eval.compare",
            "--candidate",
            f"/app/reports/eval/{KEY}-candidate/metrics.json",
            "--production",
            f"/app/reports/eval/{KEY}-production/metrics.json",
        ],
        do_xcom_push=True,
    )

    @task.branch
    def decide(verdict: str) -> str:
        result = json.loads(verdict)
        log.info("verdict: %s", json.dumps(result))
        return "promote" if result["promote"] else "keep_production"

    promote = app_task(
        "promote",
        [
            "legalrag.index.alias",
            "promote",
            "--from-alias",
            CANDIDATE_ALIAS,
            "--to-alias",
            PRODUCTION_ALIAS,
        ],
    )
    keep = EmptyOperator(task_id="keep_production")

    @task(trigger_rule="none_failed_min_one_success")
    def archive(key: str) -> str:
        """Move the processed PDF aside so the sensor waits for the next one."""
        done = INCOMING / "done"
        done.mkdir(exist_ok=True)
        target = done / f"{key}.pdf"
        shutil.move(INCOMING / PDF, target)
        return str(target)

    key = run_key()
    wait >> key >> parse >> validate >> build >> [eval_candidate, eval_production] >> compare
    decide(compare.output) >> [promote, keep]
    [promote, keep] >> archive(key)


reindex()
