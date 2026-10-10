"""Load test for the /ask API: a realistic question mix, per-user API keys, TTFT for streams.

    uv run --group load locust -f loadtest/locustfile.py --headless -u 50 -r 5 -t 4m \
        --host http://127.0.0.1:8010 --csv reports/load/<name>

Traffic mix (per user): 80 % /ask (golden questions, Arabic and English), 15 % /ask?stream=true,
5 % /metadata. Think time 2-5 s: a person reads an answer before asking the next question (it also
keeps one user under the 30/min rate limit, so 429s mean the limiter fired on a real burst).
Every simulated user sends its own X-API-Key, the way separate clients would.

Besides Locust's own numbers, two custom entries:
- "SSE ttft": time from sending the request to the first token event (what the user feels);
- 429 responses are reported under their own name, so a rate-limited request is not hidden
  among real errors.
"""

from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

from locust import HttpUser, between, events, task

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_set.jsonl"
QUESTIONS = [
    row["question"]
    for row in map(json.loads, GOLDEN.read_text(encoding="utf-8").splitlines())
    if row["category"] in {"in_scope", "explicit_ref", "repealed"}
]
_user_ids = itertools.count(1)


class LegalQuestionUser(HttpUser):
    wait_time = between(2, 5)

    def on_start(self) -> None:
        self.client.headers["X-API-Key"] = f"loadtest-user-{next(_user_ids)}"

    def _check(self, response) -> None:
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
                if line.startswith(b"data:") and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                if b'"type": "error"' in line:
                    r.failure("error event in stream")
                    return
            r.success()
        if ttft_ms is not None:
            events.request.fire(request_type="SSE", name="ttft", response_time=ttft_ms,
                                response_length=0, exception=None, context={})  # fmt: skip

    @task(1)
    def metadata(self) -> None:
        self.client.get("/metadata", name="/metadata")
