"""Prometheus metrics for the API, served on ``GET /metrics``.

Label values come from fixed lists (anything else becomes "other") to keep cardinality bounded.
Latency buckets are dense around the SLAs (``/ask`` p95 < 5 s, TTFT p95 < 2 s). With
``PROMETHEUS_MULTIPROC_DIR`` set, ``render()`` aggregates the metric files of every worker.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
    multiprocess,
)

ENDPOINTS = {
    ("POST", "/ask"): "ask",
    ("POST", "/feedback"): "feedback",
    ("GET", "/health"): "health",
    ("GET", "/live"): "live",
    ("GET", "/metadata"): "metadata",
    ("GET", "/metrics"): "metrics",
}
STAGES = ("guard", "retrieve", "decide", "generate")
GUARDS = (
    "pii:national_id",
    "pii:phone",
    "pii:email",
    "injection:override",
    "injection:prompt_leak",
    "injection:role_play",
    "injection:delimiter",
    "gate:unanswerable",
    "citation:invalid",
    "citation:uncited",
)
DECIDERS = ("jev", "local")
BACKENDS = ("openrouter", "gemini", "vllm")
_PROMPT_VERSION = re.compile(r"v\d{1,3}")  # versions come from Langfuse config, so bound them

REQUEST_BUCKETS = (0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 13, 30)  # SLA: /ask p95 < 5 s
TTFT_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5, 10)  # SLA: p95 < 2 s
STAGE_BUCKETS = (0.001, 0.005, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10)


def _bounded(value: str, allowed: Iterable[str]) -> str:
    return value if value in allowed else "other"


def _prompt_label(version: str | None) -> str:
    return version if version and _PROMPT_VERSION.fullmatch(version) else "other"


class RagMetrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        r = registry
        if r is not None:  # process stats (Linux only); start time marks deploys in Grafana
            ProcessCollector(registry=r)
        self.requests = Counter(
            "rag_requests_total",
            "HTTP requests by endpoint and status code",
            ["endpoint", "status"],
            registry=r,
        )
        self.request_seconds = Histogram(
            "rag_request_seconds",
            "Time to response (for a stream: to its first byte)",
            ["endpoint"],
            buckets=REQUEST_BUCKETS,
            registry=r,
        )
        self.inflight = Gauge(
            "rag_inflight_requests",
            "Requests being served right now",
            multiprocess_mode="livesum",
            registry=r,
        )
        self.stage_seconds = Histogram(
            "rag_stage_seconds",
            "Time per pipeline stage",
            ["stage"],
            buckets=STAGE_BUCKETS,
            registry=r,
        )
        self.ttft = Histogram(
            "rag_ttft_seconds",
            "Streamed answers: time to the first token",
            buckets=TTFT_BUCKETS,
            registry=r,
        )
        # prompt_version per answer: a Langfuse label move changes it without a restart
        self.answers = Counter(
            "rag_answers_total",
            "Answers by outcome and served prompt",
            ["outcome", "prompt_version"],
            registry=r,
        )
        self.guardrails = Counter("rag_guardrail_total", "Guards that fired", ["guard"], registry=r)
        self.decisions = Counter(
            "rag_decisions_total",
            "Decider verdicts (pass or gated)",
            ["decider", "outcome"],
            registry=r,
        )
        self.tokens = Counter(
            "rag_llm_tokens_total",
            "LLM tokens by backend and type",
            ["backend", "type"],
            registry=r,
        )
        self.decider_cost = Counter(
            "rag_decider_cost_usd_total", "Money spent on the decider", registry=r
        )
        self.ratelimit = Counter(
            "rag_ratelimit_total", "Rate-limit verdicts", ["outcome"], registry=r
        )
        self.info = Gauge(
            "rag_info",
            "What is serving (value is always 1)",
            [
                "app_version",
                "prompt_version",
                "llm_model",
                "decider",
                "index_collection",
                "config_source",
            ],
            multiprocess_mode="max",
            registry=r,
        )

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        endpoint = ENDPOINTS.get((method, path), "other")
        self.requests.labels(endpoint, str(status)).inc()
        self.request_seconds.labels(endpoint).observe(seconds)

    def _common(
        self,
        refused: bool,
        guardrails: Iterable[str],
        usage: Mapping[str, int],
        backend: str,
        prompt_version: str | None,
    ) -> None:
        outcome = "refused" if refused else "answered"
        self.answers.labels(outcome, _prompt_label(prompt_version)).inc()
        for guard in guardrails:
            self.guardrails.labels(_bounded(guard, GUARDS)).inc()
        backend = _bounded(backend, BACKENDS)
        for kind in ("prompt", "completion"):
            if tokens := usage.get(f"{kind}_tokens", 0):
                self.tokens.labels(backend, kind).inc(tokens)

    def _stages(self, timings: Mapping[str, float | None]) -> None:
        for stage, ms in timings.items():
            if stage in STAGES and ms is not None:
                self.stage_seconds.labels(stage).observe(ms / 1000)

    def observe_answer(self, answer: Any, backend: str) -> None:
        self._stages(answer.timings_ms)
        self._common(
            answer.refused, answer.guardrails, answer.usage, backend, answer.prompt_version
        )
        if decision := answer.decision:
            outcome = "gated" if answer.gated else "pass"
            self.decisions.labels(_bounded(decision.decider, DECIDERS), outcome).inc()
            self.decider_cost.inc(decision.cost_usd)

    def observe_stream_done(self, event: Mapping[str, Any], backend: str) -> None:
        timings = event.get("timings_ms", {})
        self._stages(timings)
        if (ttft := timings.get("ttft")) is not None:
            self.ttft.observe(ttft / 1000)
        self._common(
            event.get("refused", False),
            event.get("guardrails", []),
            event.get("usage", {}),
            backend,
            event.get("prompt_version"),
        )

    def observe_stream_error(self) -> None:
        """Count a stream that failed after its 200 was sent; the middleware cannot see it."""
        self.answers.labels("error", "none").inc()

    def set_info(self, **labels: str) -> None:
        self.info.labels(**labels).set(1)


def render(registry: CollectorRegistry) -> tuple[bytes, str]:
    """The /metrics body: this process's registry, or every worker's files when multiprocess."""
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        ProcessCollector(registry=registry)  # this worker's start time, for deploy annotations
    return generate_latest(registry), CONTENT_TYPE_LATEST
