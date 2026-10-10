"""The Prometheus metric contract (written down in reports/module-5.md before this code).

Prometheus *pulls*: it scrapes ``GET /metrics`` every 15 s and stores what it reads. So the API
only keeps counters and histograms in memory; it never pushes anything.

Rules this module enforces:
- **Bounded labels.** Every label value comes from a fixed list (endpoint names, stages, guard
  names, deciders); anything else becomes "other". A raw path, a question or a request id as a
  label would create a new time series per value and take Prometheus down (cardinality).
- **Buckets around the SLA** (``/ask`` p95 < 5 s, time to first token p95 < 2 s): the histogram
  has fine buckets where the decision is made and coarse ones elsewhere.
- **Multiprocess-ready.** With several uvicorn workers each process writes its numbers to files
  in ``PROMETHEUS_MULTIPROC_DIR`` and ``render()`` adds them up; with one worker it reads memory.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

ENDPOINTS = {("POST", "/ask"): "ask", ("POST", "/feedback"): "feedback",
             ("GET", "/health"): "health", ("GET", "/live"): "live",
             ("GET", "/metadata"): "metadata", ("GET", "/metrics"): "metrics"}  # fmt: skip
STAGES = ("guard", "retrieve", "decide", "generate")
GUARDS = ("pii:national_id", "pii:phone", "pii:email", "injection:override",
          "injection:prompt_leak", "injection:role_play", "injection:delimiter",
          "gate:unanswerable", "citation:invalid", "citation:uncited")  # fmt: skip
DECIDERS = ("jev", "local")
BACKENDS = ("openrouter", "gemini", "vllm")

REQUEST_BUCKETS = (0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 13, 30)  # SLA: /ask p95 < 5 s
TTFT_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5, 10)  # SLA: p95 < 2 s
STAGE_BUCKETS = (0.001, 0.005, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10)


def _bounded(value: str, allowed: Iterable[str]) -> str:
    return value if value in allowed else "other"


class RagMetrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        r = registry
        self.requests = Counter("rag_requests_total", "HTTP requests by endpoint and status code",
                                ["endpoint", "status"], registry=r)  # fmt: skip
        self.request_seconds = Histogram("rag_request_seconds", "Time to response (for a stream: "
                                         "to its first byte)", ["endpoint"],
                                         buckets=REQUEST_BUCKETS, registry=r)  # fmt: skip
        self.inflight = Gauge("rag_inflight_requests", "Requests being served right now",
                              multiprocess_mode="livesum", registry=r)  # fmt: skip
        self.stage_seconds = Histogram("rag_stage_seconds", "Time per pipeline stage", ["stage"],
                                       buckets=STAGE_BUCKETS, registry=r)  # fmt: skip
        self.ttft = Histogram("rag_ttft_seconds", "Streamed answers: time to the first token",
                              buckets=TTFT_BUCKETS, registry=r)  # fmt: skip
        self.answers = Counter("rag_answers_total", "Answers by outcome", ["outcome"], registry=r)
        self.guardrails = Counter("rag_guardrail_total", "Guards that fired", ["guard"], registry=r)
        self.decisions = Counter("rag_decisions_total", "Decider verdicts (pass or gated)",
                                 ["decider", "outcome"], registry=r)  # fmt: skip
        self.tokens = Counter("rag_llm_tokens_total", "LLM tokens by backend and type",
                              ["backend", "type"], registry=r)  # fmt: skip
        self.decider_cost = Counter("rag_decider_cost_usd_total", "Money spent on the decider",
                                    registry=r)  # fmt: skip
        self.ratelimit = Counter("rag_ratelimit_total", "Rate-limit verdicts", ["outcome"],
                                 registry=r)  # fmt: skip
        self.info = Gauge("rag_info", "What is serving (value is always 1)",
                          ["app_version", "prompt_version", "llm_model", "decider",
                           "index_collection", "config_source"],
                          multiprocess_mode="max", registry=r)  # fmt: skip

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        endpoint = ENDPOINTS.get((method, path), "other")
        self.requests.labels(endpoint, str(status)).inc()
        self.request_seconds.labels(endpoint).observe(seconds)

    def _common(self, refused: bool, guardrails: Iterable[str], usage: Mapping[str, int],
                backend: str) -> None:  # fmt: skip
        self.answers.labels("refused" if refused else "answered").inc()
        for guard in guardrails:
            self.guardrails.labels(_bounded(guard, GUARDS)).inc()
        backend = _bounded(backend, BACKENDS)
        for kind in ("prompt", "completion"):
            if tokens := usage.get(f"{kind}_tokens", 0):
                self.tokens.labels(backend, kind).inc(tokens)

    def observe_answer(self, answer: Any, backend: str) -> None:
        for stage, ms in answer.timings_ms.items():
            if stage in STAGES:
                self.stage_seconds.labels(stage).observe(ms / 1000)
        self._common(answer.refused, answer.guardrails, answer.usage, backend)
        if decision := answer.decision:
            outcome = "gated" if answer.gated else "pass"
            self.decisions.labels(_bounded(decision.decider, DECIDERS), outcome).inc()
            self.decider_cost.inc(decision.cost_usd)

    def observe_stream_done(self, event: Mapping[str, Any], backend: str) -> None:
        if (ttft := event.get("timings_ms", {}).get("ttft")) is not None:
            self.ttft.observe(ttft / 1000)
        self._common(event.get("refused", False), event.get("guardrails", []),
                     event.get("usage", {}), backend)  # fmt: skip

    def set_info(self, **labels: str) -> None:
        self.info.labels(**labels).set(1)


def render(registry: CollectorRegistry) -> tuple[bytes, str]:
    """The /metrics body: this process's registry, or every worker's files when multiprocess."""
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry), CONTENT_TYPE_LATEST
