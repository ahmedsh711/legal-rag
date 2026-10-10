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

from legalrag.generation import PROMPT_VERSION
from legalrag.logging_conf import get_logger
from legalrag.settings import Settings

log = get_logger(__name__)

RUNTIME_KEYS = ("decider_backend", "retrieval_mode", "retrieve_top_n", "rerank_keep_top",
                "gate_threshold")  # fmt: skip


class ConfigMismatchError(RuntimeError):
    """The registered config was evaluated with something this code does not serve."""


def fetch_config(
    tracking_uri: str, name: str, alias: str, transport: httpx.BaseTransport | None = None
) -> tuple[dict[str, Any], str]:
    """(config, version) behind ``models:/name@alias``."""
    with httpx.Client(base_url=tracking_uri, timeout=5, transport=transport) as client:
        r = client.get("/api/2.0/mlflow/registered-models/alias", params={"name": name,
                                                                          "alias": alias})  # fmt: skip
        r.raise_for_status()
        version = r.json()["model_version"]
        art = client.get("/get-artifact", params={"path": "rag_config.json",
                                                  "run_uuid": version["run_id"]})  # fmt: skip
        art.raise_for_status()
        return art.json(), str(version["version"])


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
    if config.get("prompt_version", PROMPT_VERSION) != PROMPT_VERSION:
        raise ConfigMismatchError(
            f"{name}@{alias} was evaluated with prompt {config.get('prompt_version')!r}, "
            f"this code serves {PROMPT_VERSION!r}"
        )
    update = {k: config[k] for k in RUNTIME_KEYS if k in config}
    log.info("config_from_registry", model=name, alias=alias, version=version, **update)
    return settings.model_copy(update=update), f"{name}@{alias} (v{version})"
