"""Shared settings for the legal-rag DAGs: every task runs in the application image."""

from __future__ import annotations

import os
from typing import Any

from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# path of the repository as the Docker daemon sees it (on Docker Desktop for Windows:
# /run/desktop/mnt/host/c/...), because task containers are started by the host daemon
HOST_DIR = os.environ.get("LEGALRAG_HOST_DIR", "/opt/legal-rag")
IMAGE = os.environ.get("LEGALRAG_IMAGE", "ahmedshobaki/legal-rag-api:dev")
DOCKER_URL = os.environ.get("DOCKER_URL", "unix://var/run/docker.sock")
NETWORK = os.environ.get("LEGALRAG_NETWORK", "legal-rag_default")

PRODUCTION_ALIAS = "articles"
CANDIDATE_ALIAS = "articles_candidate"

DEFAULT_ARGS = {"owner": "legal-rag", "retries": 1}


def app_task(
    task_id: str,
    command: list[str],
    alias: str = PRODUCTION_ALIAS,
    extra_mounts: list[Mount] | None = None,
    **kwargs: Any,
) -> DockerOperator:
    """A pipeline step run as a container of the application image."""
    mounts = [
        Mount("/app/data", f"{HOST_DIR}/data", type="bind"),
        Mount("/app/reports", f"{HOST_DIR}/reports/airflow", type="bind"),
        Mount("/app/params.yaml", f"{HOST_DIR}/params.yaml", type="bind", read_only=True),
        Mount("/home/app/.cache/huggingface", "legal-rag_hf_cache", type="volume"),
        *(extra_mounts or []),
    ]
    return DockerOperator(
        task_id=task_id,
        image=IMAGE,
        entrypoint=["python", "-m"],
        command=command,
        docker_url=DOCKER_URL,
        network_mode=NETWORK,
        mounts=mounts,
        environment={"QDRANT_URL": "http://qdrant:6333", "QDRANT_COLLECTION_ALIAS": alias},
        user="0",  # bind mounts on Docker Desktop are writable by root only
        mount_tmp_dir=False,
        auto_remove="success",
        **kwargs,
    )
