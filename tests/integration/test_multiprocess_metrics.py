"""Several uvicorn workers = several processes: /metrics must add them up, not show one of them.

prometheus_client decides at import time whether values live in memory or in files under
PROMETHEUS_MULTIPROC_DIR, so the "workers" here are real child processes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

WORKER = textwrap.dedent(
    """
    from prometheus_client import CollectorRegistry
    from legalrag.observability.metrics import RagMetrics
    m = RagMetrics(CollectorRegistry())
    for _ in range({n}):
        m.observe_request("POST", "/ask", 200, 0.4)
    m.guardrails.labels("pii:phone").inc()
    """
)


def test_metrics_from_two_worker_processes_are_summed(tmp_path):
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    for n in (3, 4):  # two workers, each served some requests
        subprocess.run([sys.executable, "-c", WORKER.format(n=n)], env=env, check=True)
    render = subprocess.run(
        [sys.executable, "-c",
         "from prometheus_client import CollectorRegistry;"
         "from legalrag.observability.metrics import render;"
         "print(render(CollectorRegistry())[0].decode())"],
        env=env, check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    assert 'rag_requests_total{endpoint="ask",status="200"} 7.0' in render
    assert 'rag_guardrail_total{guard="pii:phone"} 2.0' in render
    assert 'rag_request_seconds_count{endpoint="ask"} 7.0' in render
