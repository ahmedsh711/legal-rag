"""Load test for the /ask API: a realistic question mix, one API key per user, TTFT for streams.

    export LOADTEST_API_KEYS=$(seq -f "loadtest-%g" 1 100 | paste -sd, -)   # issued keys
    docker compose --env-file .env -f docker/compose.yaml -f docker/compose.vllm.yaml \
        -f docker/compose.loadtest.yaml --profile core --profile llm up -d
    uv run --group load locust -f loadtest/locustfile.py --headless -u 50 -r 5 -t 4m \
        --host http://127.0.0.1:8010 --csv reports/load/<name>

Traffic mix (per user): 80 % /ask (golden questions, Arabic and English), 15 % /ask?stream=true,
5 % /metadata. Think time 2-5 s: a person reads an answer before asking the next question (it also
keeps one user under the 30/min rate limit, so a 429 means the limiter fired on a real burst).
Every simulated user sends its own issued key (LOADTEST_API_KEYS, also given to the API as
API_KEYS): the API only gives a bucket of its own to keys it issued, everyone else shares a
bucket per IP, and all Locust users come from one IP.

Besides Locust's own numbers:
- "SSE ttft": time from sending the request to the first token event (what the user feels);
- "STAGE <name>": the per-stage timings the API returns in every /ask answer (guard, retrieve,
  generate, total), so a bottleneck claim names the stage whose time grows with load;
- failures carry their reason ("429 rate limited", "HTTP 503", "error event"), so the failure
  table separates a rate-limited request from a broken one.
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
                    events.request.fire(request_type="STAGE", name=stage, response_time=ms,
                                        response_length=0, exception=None, context={})  # fmt: skip

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
            events.request.fire(request_type="SSE", name="ttft", response_time=ttft_ms,
                                response_length=0, exception=None, context={})  # fmt: skip

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
