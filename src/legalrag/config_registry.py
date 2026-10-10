"""Load the serving config from the MLflow model registry by alias, over plain REST."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import ValidationError

from legalrag.generation import PROMPT_VERSION
from legalrag.logging_conf import get_logger
from legalrag.settings import Settings

log = get_logger(__name__)

RUNTIME_KEYS = (
    "decider_backend",
    "retrieval_mode",
    "retrieve_top_n",
    "rerank_keep_top",
    "gate_threshold",
)


class ConfigMismatchError(RuntimeError):
    """The registered config was evaluated with something this code does not serve."""


def fetch_run_config(
    tracking_uri: str, run_id: str, transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    """Fetch a run's rag_config.json through the tracking server.

    ``mlflow.artifacts.download_artifacts`` returns a presigned MinIO URL that only resolves
    inside Docker."""
    with httpx.Client(base_url=tracking_uri, timeout=10, transport=transport) as client:
        art = client.get("/get-artifact", params={"path": "rag_config.json", "run_uuid": run_id})
        art.raise_for_status()
        return art.json()


def fetch_config(
    tracking_uri: str, name: str, alias: str, transport: httpx.BaseTransport | None = None
) -> tuple[dict[str, Any], str]:
    """Return (config, version) behind ``models:/name@alias``."""
    with httpx.Client(base_url=tracking_uri, timeout=5, transport=transport) as client:
        r = client.get(
            "/api/2.0/mlflow/registered-models/alias", params={"name": name, "alias": alias}
        )
        r.raise_for_status()
        version = r.json()["model_version"]
    return fetch_run_config(tracking_uri, version["run_id"], transport), str(version["version"])


def apply_registry_config(
    settings: Settings, transport: httpx.BaseTransport | None = None
) -> tuple[Settings, str]:
    """Apply the registry's runtime knobs; also return where the config came from."""
    if settings.config_source != "mlflow":
        return settings, "env"
    name, alias = settings.mlflow_config_model_name, settings.mlflow_config_alias
    try:
        config, version = fetch_config(settings.mlflow_tracking_uri, name, alias, transport)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        # a dead tracking server must not take the API down
        log.error("config_registry_unreachable", error=str(exc)[:200])
        return settings, "env (registry unreachable)"
    served = _checked(config, settings, f"{name}@{alias} (v{version})")
    log.info(
        "config_from_registry",
        model=name,
        alias=alias,
        version=version,
        **{k: getattr(served, k) for k in RUNTIME_KEYS},
    )
    return served, f"{name}@{alias} (v{version})"


def _checked(config: Any, settings: Settings, where: str) -> Settings:
    """Reject at startup a config that was evaluated against a different prompt or decider."""
    if not isinstance(config, dict):
        raise ConfigMismatchError(f"{where}: rag_config.json is not a JSON object")
    if config.get("prompt_version") != PROMPT_VERSION:
        raise ConfigMismatchError(
            f"{where} was evaluated with prompt "
            f"{config.get('prompt_version')!r}, this code serves "
            f"{PROMPT_VERSION!r}"
        )
    expected = {"jev": settings.jev_model, "local": settings.reranker_model}
    backend = config.get("decider_backend", settings.decider_backend)
    if config.get("decider_model") and config["decider_model"] != expected.get(backend):
        # the gate threshold was tuned for one decider model
        raise ConfigMismatchError(
            f"{where}: decider model {config['decider_model']!r} "
            f"!= {expected.get(backend)!r} in this environment"
        )
    update = {k: config[k] for k in RUNTIME_KEYS if k in config}
    try:  # not model_copy: it skips validation
        return Settings.model_validate({**settings.model_dump(), **update})
    except ValidationError as exc:
        raise ConfigMismatchError(f"{where}: invalid value: {exc.errors()[0]['msg']}") from exc
