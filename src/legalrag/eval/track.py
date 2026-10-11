"""MLflow: one run per evaluation, and the RAG config versioned in the model registry.

Runs carry the config as params, metrics overall and per language, and lineage tags (git SHA,
articles.json md5, Qdrant collection). A chosen config is registered as a small pyfunc model
holding ``rag_config.json`` and promoted by moving the ``production`` alias.
"""

from __future__ import annotations

import argparse
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
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
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

    # MLflow's signature inference expects list[...] hints
    def predict(self, context: Any, model_input: list[str], params: Any = None) -> list[str]:
        return [json.dumps(self.config)]


def register_config(config: Mapping[str, Any], name: str, alias: str | None = None) -> str:
    """Register ``config`` as a new version of ``name`` in the active run; returns the version."""
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


def copy_config(config: Mapping[str, Any], name: str, alias: str, source: str) -> str:
    """Register a config taken from another tracking server here, under the same alias."""
    mlflow.set_experiment("legal-rag-config")
    with mlflow.start_run(run_name=f"{name}@{alias} copied"):
        mlflow.set_tag("copied_from", source)
        mlflow.log_params({k: v for k, v in config.items() if not isinstance(v, dict | list)})
        mlflow.log_dict(dict(config), CONFIG_FILE)  # where the API reads it
        return register_config(config, name, alias)


def load_config(name: str, alias: str = "production") -> dict[str, Any]:
    """The config behind ``models:/name@alias`` without unpickling anything."""
    path = mlflow.artifacts.download_artifacts(f"models:/{name}@{alias}/artifacts/{CONFIG_FILE}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    """Promote an evaluated run: register its own rag_config.json, optionally move an alias.

    uv run python -m legalrag.eval.track register --run-id <id> --alias production
    """
    parser = argparse.ArgumentParser(description="MLflow config registry")
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="register a run's rag_config.json")
    reg.add_argument("--run-id", required=True)
    reg.add_argument("--alias", default=None, help="e.g. production (omit to only register)")
    cp = sub.add_parser("copy", help="copy name@alias from another server, e.g. local -> DagsHub")
    cp.add_argument("--from", dest="source", required=True, help="source tracking server URL")
    cp.add_argument("--alias", default="production")
    args = parser.parse_args(argv)

    from legalrag.logging_conf import configure_logging, get_logger
    from legalrag.settings import get_settings

    settings = get_settings()
    configure_logging(settings.log_level)
    uri = settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    name = settings.mlflow_config_model_name
    if args.command == "copy":
        from legalrag.config_registry import fetch_config

        config, version = fetch_config(args.source, name, args.alias)
        new = copy_config(config, name, args.alias, source=f"{args.source} v{version}")
        get_logger(__name__).info("config_copied", model=name, version=new, alias=args.alias)
        return
    mode = MlflowClient().get_run(args.run_id).data.tags.get("eval_mode")
    if mode != "end_to_end":  # a retrieval-only run never measured an answer
        raise SystemExit(f"run {args.run_id} is {mode!r}; only end_to_end runs can be promoted")
    if uri.startswith("http"):  # tracking server, see fetch_run_config
        from legalrag.config_registry import fetch_run_config

        config = fetch_run_config(uri, args.run_id)
    else:  # a local store (tests)
        path = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path=CONFIG_FILE)
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    with mlflow.start_run(run_id=args.run_id):  # the version links back to the evaluated run
        version = register_config(config, settings.mlflow_config_model_name, args.alias)
    get_logger(__name__).info(
        "config_registered",
        model=settings.mlflow_config_model_name,
        version=version,
        alias=args.alias,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
