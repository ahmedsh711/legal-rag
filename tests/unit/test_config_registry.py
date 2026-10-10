"""Loading the RAG config by alias from the MLflow registry (REST, no mlflow library)."""

import httpx
import pytest

from legalrag.config_registry import (
    ConfigMismatchError,
    apply_registry_config,
    fetch_config,
    fetch_run_config,
)
from legalrag.generation import PROMPT_VERSION
from legalrag.settings import Settings

CONFIG = {"decider_backend": "jev", "retrieval_mode": "dense", "gate_threshold": 0.5,
          "rerank_keep_top": 5, "prompt_version": PROMPT_VERSION, "llm_model": "whatever"}  # fmt: skip


def mlflow_server(config=CONFIG, alias_status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/2.0/mlflow/registered-models/alias":
            assert request.url.params["name"] == "legal-rag-config"
            assert request.url.params["alias"] == "production"
            return httpx.Response(alias_status, json={"model_version": {"version": "3",
                                                                        "run_id": "r1"}})  # fmt: skip
        if request.url.path == "/get-artifact":
            assert request.url.params["run_uuid"] == "r1"
            return httpx.Response(200, json=config)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def settings(**kw) -> Settings:
    return Settings(_env_file=None, config_source="mlflow", **kw)


def test_fetch_config_follows_alias_to_the_run_artifact():
    config, version = fetch_config("http://mlflow:5000", "legal-rag-config", "production",
                                   transport=mlflow_server())  # fmt: skip
    assert config == CONFIG and version == "3"


def test_fetch_run_config_reads_the_run_artifact_through_the_server():
    config = fetch_run_config("http://mlflow:5000", "r1", transport=mlflow_server())
    assert config == CONFIG


def test_only_runtime_knobs_are_applied():
    s, source = apply_registry_config(settings(), transport=mlflow_server())
    assert (s.decider_backend, s.retrieval_mode, s.gate_threshold) == ("jev", "dense", 0.5)
    assert s.llm_model == Settings(_env_file=None).llm_model  # models are not overridden
    assert source == "legal-rag-config@production (v3)"


def test_a_config_evaluated_with_another_prompt_is_refused():
    other = {**CONFIG, "prompt_version": "v0"}
    with pytest.raises(ConfigMismatchError, match="prompt"):
        apply_registry_config(settings(), transport=mlflow_server(config=other))


def test_unreachable_registry_keeps_env_settings():
    s, source = apply_registry_config(settings(decider_backend="none"),
                                      transport=mlflow_server(alias_status=503))  # fmt: skip
    assert s.decider_backend == "none" and source.startswith("env")


def test_env_source_does_not_call_mlflow():
    s, source = apply_registry_config(Settings(_env_file=None), transport=None)
    assert source == "env"
