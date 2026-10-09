"""MLflow: one run per evaluation, and the RAG config versioned in the model registry.

A run records *what was tried* (params: models, decider, top-k, prompt version), *how it did*
(metrics, overall and per language) and *exactly which data and code* (tags: git SHA, the md5 of
articles.json, the Qdrant collection). That is the lineage chain git SHA -> DVC hash -> MLflow run.

The winning config is registered as a tiny pyfunc model holding ``rag_config.json`` and promoted by
moving the alias ``production`` ("load by alias, never by path").
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel

CONFIG_FILE = "rag_config.json"


def flatten_metrics(summary: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """{"all": {"mrr": 0.8}, "ar": {...}} -> {"all.mrr": 0.8, "ar.mrr": ...}; drops None."""
    return {
        f"{group}.{name}": float(value)
        for group, metrics in summary.items()
        for name, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True, timeout=10)  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def log_eval_run(
    experiment: str,
    run_name: str,
    params: Mapping[str, Any],
    summary: Mapping[str, Mapping[str, Any]],
    tags: Mapping[str, str],
    artifacts: Iterable[str | Path] = (),
) -> str:
    """Log one evaluation as an MLflow run; returns the run id."""
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags={"git_sha": git_sha(), **tags}) as run:
        mlflow.log_params(dict(params))
        mlflow.log_metrics(flatten_metrics(summary))
        mlflow.log_dict(dict(params), CONFIG_FILE)  # the exact config, as a file
        for path in artifacts:
            mlflow.log_artifact(str(path))
        return run.info.run_id


class RagConfig(PythonModel):
    """A registered "model" that is only a config: lets the registry version and alias it."""

    def load_context(self, context: Any) -> None:
        self.config = json.loads(Path(context.artifacts["config"]).read_text(encoding="utf-8"))

    # list[...] type hints: what MLflow's signature inference expects
    def predict(self, context: Any, model_input: list[str], params: Any = None) -> list[str]:
        return [json.dumps(self.config)]


def register_config(config: Mapping[str, Any], name: str, alias: str | None = None) -> str:
    """Register ``config`` as a new version of ``name`` (inside the active run); optionally
    point ``alias`` at it. Returns the version number."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / CONFIG_FILE
        path.write_text(json.dumps(dict(config), indent=2), encoding="utf-8", newline="\n")
        info = mlflow.pyfunc.log_model(
            name="rag_config",
            python_model=RagConfig(),
            artifacts={"config": str(path)},
            registered_model_name=name,
            pip_requirements=[],
        )
    version = str(info.registered_model_version)
    if alias:
        MlflowClient().set_registered_model_alias(name, alias, version)
    return version


def load_config(name: str, alias: str = "production") -> dict[str, Any]:
    """The config behind ``models:/name@alias`` without unpickling anything."""
    path = mlflow.artifacts.download_artifacts(f"models:/{name}@{alias}/artifacts/{CONFIG_FILE}")
    return json.loads(Path(path).read_text(encoding="utf-8"))
