"""Settings: defaults, env overrides and values derived from the chosen backend."""

from legalrag.settings import Settings


def test_defaults_are_sane_without_env(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    s = Settings(_env_file=None)
    assert s.llm_backend == "openrouter"
    assert s.decider_backend == "none"
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


def test_gemini_backend_for_generation_and_judge(monkeypatch):
    gem, orr = "fake-gemini", "fake-openrouter"  # dummy test values, not secrets
    s = Settings(
        _env_file=None,
        llm_backend="gemini",
        judge_backend="gemini",
        gemini_api_key=gem,
        openrouter_api_key=orr,
    )
    assert s.active_llm_base_url == s.gemini_base_url and s.active_llm_model == s.gemini_model
    assert s.active_llm_api_key == gem
    assert s.judge_base_url == s.gemini_base_url and s.judge_api_key == gem
    default = Settings(_env_file=None, openrouter_api_key=orr)
    assert default.judge_base_url == default.llm_base_url and default.judge_api_key == orr


def test_secrets_are_not_printed():
    s = Settings(_env_file=None, openrouter_api_key="sk-secret")
    assert "sk-secret" not in repr(s)
