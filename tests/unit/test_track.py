"""MLflow logging and config registry against a throwaway local tracking store."""

import pytest

mlflow = pytest.importorskip("mlflow")

from legalrag.eval.track import (  # noqa: E402
    flatten_metrics,
    load_config,
    log_eval_run,
    register_config,
)


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    # MLflow keeps a global tracking URI, active experiment and ./mlruns; isolate all three
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("legal-rag-test")
    yield uri


def test_flatten_metrics_prefixes_groups_and_drops_none():
    flat = flatten_metrics({"all": {"mrr": 0.8, "n": 56, "x": None}, "ar": {"mrr": 0.7}})
    assert flat == {"all.mrr": 0.8, "all.n": 56.0, "ar.mrr": 0.7}


def test_eval_run_records_params_metrics_tags_and_config(tracking, tmp_path):
    preds = tmp_path / "predictions.jsonl"
    preds.write_text("{}\n", encoding="utf-8")
    run_id = log_eval_run(
        "legal-rag-test",
        "baseline",
        params={"decider": "none", "context_size": 5},
        summary={"all": {"mrr": 0.8}},
        tags={"articles_md5": "abc"},
        artifacts=[preds],
    )
    run = mlflow.get_run(run_id)
    assert run.data.params == {"decider": "none", "context_size": "5"}
    assert run.data.metrics["all.mrr"] == 0.8
    assert run.data.tags["articles_md5"] == "abc" and "git_sha" in run.data.tags
    names = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id)}
    assert {"predictions.jsonl", "rag_config.json"} <= names


def test_register_cli_promotes_exactly_the_evaluated_config(tracking, monkeypatch):
    from legalrag.eval.track import main
    from legalrag.settings import get_settings

    run_id = log_eval_run(
        "legal-rag-test",
        "jev",
        params={"decider_backend": "jev"},
        summary={"all": {"mrr": 1.0}},
        tags={"eval_mode": "end_to_end"},
    )
    retrieval_only = log_eval_run(
        "legal-rag-test",
        "ret",
        params={"decider_backend": "jev"},
        summary={},
        tags={"eval_mode": "retrieval_only"},
    )
    monkeypatch.setenv("MLFLOW_CONFIG_MODEL_NAME", "rag-config-cli")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit, match="end_to_end"):  # never measured an answer
            main(["register", "--run-id", retrieval_only, "--alias", "production"])
        main(["register", "--run-id", run_id, "--alias", "production"])
    finally:
        get_settings.cache_clear()
    assert load_config("rag-config-cli", "production") == {"decider_backend": "jev"}


def test_config_is_loaded_by_alias(tracking):
    with mlflow.start_run():
        v1 = register_config({"decider": "none"}, "rag-config-test", alias="production")
    with mlflow.start_run():
        register_config({"decider": "jev"}, "rag-config-test")  # new version, not promoted
    assert load_config("rag-config-test", "production") == {"decider": "none"}
    mlflow.MlflowClient().set_registered_model_alias("rag-config-test", "production", "2")
    assert load_config("rag-config-test", "production") == {"decider": "jev"}  # promotion
    assert v1 == "1"
