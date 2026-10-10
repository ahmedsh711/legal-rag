"""The serving config comes from the MLflow registry: "load by alias, never by path".

The best evaluated config is registered as ``legal-rag-config`` and the alias ``production``
points at one version. At startup (``CONFIG_SOURCE=mlflow``) the API asks MLflow which version
that is and reads its ``rag_config.json``. Promoting or rolling back a config = moving the alias;
no rebuild, no new image. Two REST calls, so the API image does not need the mlflow library.

Only runtime knobs are taken from the registry. Model identities (LLM, embedding) stay in the
environment, and a config evaluated with another prompt version than the code serves is refused.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import ValidationError

from legalrag.generation import PROMPT_VERSION
from legalrag.logging_conf import get_logger
from legalrag.settings import Settings

log = get_logger(__name__)

RUNTIME_KEYS = ("decider_backend", "retrieval_mode", "retrieve_top_n", "rerank_keep_top",
                "gate_threshold")  # fmt: skip


class ConfigMismatchError(RuntimeError):
    """The registered config was evaluated with something this code does not serve."""


def fetch_run_config(
    tracking_uri: str, run_id: str, transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    """A run's rag_config.json, streamed through the tracking server.

    Not ``mlflow.artifacts.download_artifacts``: MLflow 3.17 hands the client a presigned URL
    for MinIO's in-network name (http://minio:9000), which does not resolve outside Docker."""
    with httpx.Client(base_url=tracking_uri, timeout=10, transport=transport) as client:
        art = client.get("/get-artifact", params={"path": "rag_config.json", "run_uuid": run_id})
        art.raise_for_status()
        return art.json()


def fetch_config(
    tracking_uri: str, name: str, alias: str, transport: httpx.BaseTransport | None = None
) -> tuple[dict[str, Any], str]:
    """(config, version) behind ``models:/name@alias``."""
    with httpx.Client(base_url=tracking_uri, timeout=5, transport=transport) as client:
        r = client.get("/api/2.0/mlflow/registered-models/alias", params={"name": name,
                                                                          "alias": alias})  # fmt: skip
        r.raise_for_status()
        version = r.json()["model_version"]
    return fetch_run_config(tracking_uri, version["run_id"], transport), str(version["version"])


def apply_registry_config(
    settings: Settings, transport: httpx.BaseTransport | None = None
) -> tuple[Settings, str]:
    """Settings with the registry's runtime knobs applied, and where the config came from."""
    if settings.config_source != "mlflow":
        return settings, "env"
    name, alias = settings.mlflow_config_model_name, settings.mlflow_config_alias
    try:
        config, version = fetch_config(settings.mlflow_tracking_uri, name, alias, transport)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        # the tracking server being down must not take the answering service down
        log.error("config_registry_unreachable", error=str(exc)[:200])
        return settings, "env (registry unreachable)"
    served = _checked(config, settings, f"{name}@{alias} (v{version})")
    log.info("config_from_registry", model=name, alias=alias, version=version,
             **{k: getattr(served, k) for k in RUNTIME_KEYS})  # fmt: skip
    return served, f"{name}@{alias} (v{version})"


def _checked(config: Any, settings: Settings, where: str) -> Settings:
    """Refuse (loudly, at startup) a config this code cannot serve faithfully."""
    if not isinstance(config, dict):
        raise ConfigMismatchError(f"{where}: rag_config.json is not a JSON object")
    if config.get("prompt_version") != PROMPT_VERSION:
        raise ConfigMismatchError(f"{where} was evaluated with prompt "
                                  f"{config.get('prompt_version')!r}, this code serves "
                                  f"{PROMPT_VERSION!r}")  # fmt: skip
    expected = {"jev": settings.jev_model, "local": settings.reranker_model}
    backend = config.get("decider_backend", settings.decider_backend)
    if config.get("decider_model") and config["decider_model"] != expected.get(backend):
        # the gate threshold was tuned for one decider model
        raise ConfigMismatchError(f"{where}: decider model {config['decider_model']!r} "
                                  f"!= {expected.get(backend)!r} in this environment")  # fmt: skip
    update = {k: config[k] for k in RUNTIME_KEYS if k in config}
    try:  # model_copy would skip validation: a typo like "Hybrid" would silently serve sparse
        return Settings.model_validate({**settings.model_dump(), **update})
    except ValidationError as exc:
        raise ConfigMismatchError(f"{where}: invalid value: {exc.errors()[0]['msg']}") from exc
