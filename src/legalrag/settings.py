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
    api_port: int = 8000

    # --- generation backend (OpenAI-compatible everywhere: OpenRouter or local vLLM) ---
    llm_backend: Literal["openrouter", "vllm"] = "openrouter"
    openrouter_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "qwen/qwen-2.5-72b-instruct"
    judge_model: str = "anthropic/claude-sonnet-5-5"
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    llm_timeout_s: float = 60.0
    llm_max_tokens: int = 800
    llm_temperature: float = 0.0

    # --- Jev decision model ---
    decider_backend: Literal["jev", "local"] = "jev"
    typesafe_api_key: SecretStr = SecretStr("")
    jev_model: str = "jev-1.13.0"
    jev_timeout_s: float = 2.0
    # tiers from the JEV-RAG pattern: act / second opinion / refuse
    tier_act: float = Field(0.90, ge=0, le=1)
    tier_second_opinion: float = Field(0.60, ge=0, le=1)
    gate_threshold: float = Field(0.75, ge=0, le=1)
    validate_threshold: float = Field(0.75, ge=0, le=1)
    rerank_keep_top: int = 5
    retrieve_top_n: int = 12
    jev_translate_query: bool = False

    # --- retrieval ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: Literal["cpu", "cuda"] = "cpu"
    normalization_version: str = "v1"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_alias: str = "articles"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- data paths (relative to repo root) ---
    raw_pdf_path: str = "data/raw/egyptian_civil_code.pdf"
    articles_path: str = "data/processed/articles.json"
    golden_set_path: str = "data/golden/golden_set.jsonl"

    # --- infra ---
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_config_model_name: str = "legal-rag-config"
    mlflow_config_alias: str = "production"
    langfuse_host: str = "http://localhost:3000"
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
