"""Single source of configuration.

Every path, URL, model name and threshold lives here and is read from the environment
(or a local ``.env``). Nothing else in the package hard-codes these values. This is the
course's "config via env vars with sane defaults, no hard-coded paths" rule.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- service ---
    app_name: str = "legal-rag"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"

    # --- generation backend (OpenAI-compatible everywhere: OpenRouter or local vLLM) ---
    llm_backend: Literal["openrouter", "vllm"] = "openrouter"
    openrouter_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "qwen/qwen3-235b-a22b-2507"  # strong multilingual, cheap
    # USD per million tokens for llm_model (OpenRouter /models, checked 2026-10-09); eval cost
    llm_price_in_per_m: float = 0.09
    llm_price_out_per_m: float = 0.55
    judge_model: str = (
        "anthropic/claude-haiku-5.5"  # different family from the generator (self-preference bias)
    )
    vllm_base_url: str = "http://127.0.0.1:8001/v1"
    vllm_model: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    llm_timeout_s: float = 60.0
    llm_max_tokens: int = 800
    llm_temperature: float = 0.0

    # --- decider: rerank + "can we answer?" gate (none | jev | local) ---
    decider_backend: Literal["none", "jev", "local"] = "none"
    # Jev through OpenRouter's /systemone endpoint (same key as generation; one bill)
    jev_base_url: str = "https://openrouter.ai/api/v1"
    jev_model: str = "jev-1.13"
    jev_timeout_s: float = 10.0  # first call measured 2.9 s
    gate_threshold: float = Field(0.75, ge=0, le=1)  # below this: refuse without calling the LLM
    rerank_keep_top: int = 5  # articles shown to the LLM
    retrieve_top_n: int = 12  # articles retrieved and scored by the decider

    # --- retrieval ---
    embedding_model: str = "BAAI/bge-m3"
    # pinned commit (the safetensors conversion of bge-m3); must match params.yaml -> index
    # (a public git commit sha, not a secret)
    embedding_revision: str = "9a0624b896d81da7492a910ffa53731274b6cf3d"  # pragma: allowlist secret
    embedding_device: Literal["cpu", "cuda"] = "cpu"
    query_max_length: int = 512  # questions are short; shorter max length = faster query embedding
    # 127.0.0.1, not "localhost": compose publishes ports on IPv4 loopback only, and on Windows
    # "localhost" tries IPv6 (::1) first -> every Qdrant call waited ~2 s (measured 2,069 vs 17 ms)
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection_alias: str = "articles"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # the local decider (cross-encoder)
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"  # pragma: allowlist secret

    # --- data paths (relative to repo root) ---
    raw_pdf_path: str = "data/raw/egyptian_civil_code.pdf"
    articles_path: str = "data/processed/articles.json"
    golden_set_path: str = "data/golden/golden_set.jsonl"
    feedback_path: str = "data/feedback/feedback.jsonl"

    # --- infra ---
    redis_url: str = "redis://127.0.0.1:6379/0"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_config_model_name: str = "legal-rag-config"
    mlflow_config_alias: str = "production"
    langfuse_host: str = "http://127.0.0.1:3000"
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    rate_limit_per_minute: int = 60

    @property
    def active_llm_base_url(self) -> str:
        return self.vllm_base_url if self.llm_backend == "vllm" else self.llm_base_url

    @property
    def active_llm_model(self) -> str:
        return self.vllm_model if self.llm_backend == "vllm" else self.llm_model

    @property
    def active_llm_api_key(self) -> str:
        # vLLM ignores the key but the OpenAI client requires a non-empty string
        return self.openrouter_api_key.get_secret_value() or "none"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the app reads the environment once (load-once rule)."""
    return Settings()
