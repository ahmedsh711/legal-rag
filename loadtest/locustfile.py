"""Locust load test for /ask: golden-set questions, one issued API key per user, SSE TTFT.

    export LOADTEST_API_KEYS=$(seq -f "loadtest-%g" 1 100 | paste -sd, -)
    docker compose --env-file .env -f docker/compose.yaml -f docker/compose.vllm.yaml \
        -f docker/compose.loadtest.yaml --profile core --profile llm up -d
    uv run python loadtest/sample_stats.py --out reports/load/<name>-stats.csv --seconds 490 &
    LOCUST_STEPS=5,10,20,40 LOCUST_STEP_SECONDS=120 uv run --group load python -m locust \
        -f loadtest/locustfile.py --headless --host http://127.0.0.1:8010 --csv reports/load/<name>

Then loadtest/analyze.py builds the per-step table. Use ``python -m locust``: the locust.exe
shim can hang when antivirus holds new executables.

Mix per user: 80 % /ask (Arabic and English), 15 % /ask?stream=true, 5 % /metadata, with 2-5 s
think time. That keeps a single user under the 30/min rate limit, so a 429 means a real burst.
Each user sends its own key from LOADTEST_API_KEYS (also passed to the API as API_KEYS): only
issued keys get their own rate-limit bucket, and all Locust users share one IP.

Extra entries in Locust's stats:
- "SSE ttft": time from request to the first token event;
- "STAGE <name>": per-stage timings from each /ask response (guard, retrieve, generate, total),
  to see which stage grows with load.
Failures are labelled ("429 rate limited", "HTTP 503", "error event") so rate limiting is
reported separately from errors.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from locust import HttpUser, LoadTestShape, between, events, task

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_set.jsonl"
QUESTIONS = [
    row["question"]
    for row in map(json.loads, GOLDEN.read_text(encoding="utf-8").splitlines())
    if row["category"] in {"in_scope", "explicit_ref", "repealed"}
]
KEYS = [k for k in os.environ.get("LOADTEST_API_KEYS", "").split(",") if k] or ["loadtest-1"]
_next_key = itertools.cycle(KEYS)


class LegalQuestionUser(HttpUser):
    wait_time = between(2, 5)

    def on_start(self) -> None:
        self.client.headers["X-API-Key"] = next(_next_key)

    def _check(self, response: Any) -> None:
        if response.status_code == 429:
            response.failure("429 rate limited")
        elif response.status_code == 0:  # no response at all: keep the client-side exception
            response.failure(f"no response: {getattr(response, 'error', None)!r}")
        elif response.status_code != 200:
            response.failure(f"HTTP {response.status_code}")

    @task(16)
    def ask(self) -> None:
        question = random.choice(QUESTIONS)  # noqa: S311 - traffic mix, not security
        with self.client.post(
            "/ask", json={"question": question}, name="/ask", catch_response=True
        ) as r:
            self._check(r)
            if r.status_code == 200:
                for stage, ms in r.json().get("timings_ms", {}).items():
                    events.request.fire(
                        request_type="STAGE",
                        name=stage,
                        response_time=ms,
                        response_length=0,
                        exception=None,
                        context={},
                    )

    @task(3)
    def ask_stream(self) -> None:
        question = random.choice(QUESTIONS)  # noqa: S311
        t0 = time.perf_counter()
        ttft_ms = None
        with self.client.post(
            "/ask?stream=true",
            json={"question": question},
            name="/ask?stream",
            stream=True,
            catch_response=True,
        ) as r:
            if r.status_code != 200:
                self._check(r)
                return
            for line in r.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                event = json.loads(line[5:])
                if event["type"] == "token" and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                elif event["type"] == "error":
                    r.failure("error event")
                    return
            r.success()
        if ttft_ms is not None:
            events.request.fire(
                request_type="SSE",
                name="ttft",
                response_time=ttft_ms,
                response_length=0,
                exception=None,
                context={},
            )

    @task(1)
    def metadata(self) -> None:
        self.client.get("/metadata", name="/metadata")


if os.environ.get("LOCUST_STEPS"):  # e.g. "5,10,20,40": a stepped ramp to find the knee

    class StepLoad(LoadTestShape):
        """Hold each user count for LOCUST_STEP_SECONDS (default 120), then stop."""

        steps = [int(n) for n in os.environ["LOCUST_STEPS"].split(",")]
        hold = int(os.environ.get("LOCUST_STEP_SECONDS", "120"))

        def tick(self) -> tuple[int, float] | None:
            step = int(self.get_run_time() // self.hold)
            return (self.steps[step], 5.0) if step < len(self.steps) else None
