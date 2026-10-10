"""Fail if a DAG does not import or its shape changed. Runs inside the Airflow image:

docker run --rm -v "$PWD/airflow:/opt/airflow/project:ro" -e PYTHONPATH=/opt/airflow/project/dags \
    apache/airflow:3.3.2-python3.12 python /opt/airflow/project/check_dags.py
"""

from __future__ import annotations

import sys

from airflow.dag_processing.dagbag import DagBag

REINDEX_TASKS = """wait_for_pdf run_key parse validate build_candidate eval_candidate
    eval_production compare decide promote keep_production archive"""
EXPECTED = {
    "reindex": set(REINDEX_TASKS.split()),
    "monitoring": {"drift_job", "route", "dispatch_retraining", "no_action"},
}


def main() -> int:
    bag = DagBag("/opt/airflow/project/dags")
    problems = [f"{path}: {error}" for path, error in bag.import_errors.items()]
    for dag_id, tasks in EXPECTED.items():
        dag = bag.dags.get(dag_id)
        if dag is None:
            problems.append(f"missing DAG {dag_id}")
        elif set(dag.task_ids) != tasks:
            problems.append(f"{dag_id}: tasks {sorted(dag.task_ids)}")
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    sys.stdout.write(f"{len(bag.dags)} DAGs, {len(problems)} problems\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
