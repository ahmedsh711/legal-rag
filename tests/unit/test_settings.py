"""Settings are the single source of config (course rule: no hard-coded paths/thresholds)."""

from legalrag.settings import Settings


def test_defaults_are_sane_without_env(monkeypatch):
    # Arrange: no .env, no env vars
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    # Act
    s = Settings(_env_file=None)
    # Assert
    assert s.llm_backend == "openrouter"
    assert s.decider_backend == "none"  # off until the Phase 3 ablation picks one
    assert s.embedding_model == "BAAI/bge-m3"
    assert 0 < s.gate_threshold <= 1
    assert s.rerank_keep_top <= s.retrieve_top_n


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.setenv("GATE_THRESHOLD", "0.6")
    s = Settings(_env_file=None)
    assert s.llm_backend == "vllm"
    assert s.gate_threshold == 0.6


def test_active_llm_base_url_follows_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm:8000/v1")
    s = Settings(_env_file=None)
    assert s.active_llm_base_url == "http://vllm:8000/v1"
    assert s.active_llm_model == s.vllm_model


def test_secrets_are_not_printed():
    s = Settings(_env_file=None, openrouter_api_key="sk-secret")
    assert "sk-secret" not in repr(s)
