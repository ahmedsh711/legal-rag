"""Sample container resources, the vLLM queue and the GPU every few seconds during a load test.

    uv run python loadtest/sample_stats.py --out reports/load/<name>-stats.csv --seconds 600

Writes one CSV row per container per sample: CPU %, memory, vLLM running/waiting requests,
KV-cache usage, GPU utilisation and memory. Read together with Locust's per-stage timings to
find both the stage that slows down and the resource that saturates.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

CONTAINERS = ["legal-rag-api-1", "legal-rag-vllm-1", "legal-rag-qdrant-1", "legal-rag-redis-1"]
VLLM_METRICS = "http://127.0.0.1:8001/metrics"
GAUGES = {  # vLLM Prometheus gauges (names as exported by v0.30)
    "vllm_running": r"^vllm:num_requests_running\{[^}]*\}\s+([\d.e+-]+)",
    "vllm_waiting": r"^vllm:num_requests_waiting\{[^}]*\}\s+([\d.e+-]+)",
    "vllm_kv_cache": r"^vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.e+-]+)",
}


def docker_stats() -> list[dict[str, str]]:
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}", *CONTAINERS],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def vllm_gauges() -> dict[str, float | None]:
    try:
        text = urllib.request.urlopen(VLLM_METRICS, timeout=3).read().decode()  # noqa: S310
    except OSError:
        return dict.fromkeys(GAUGES)
    found = {}
    for name, pattern in GAUGES.items():
        m = re.search(pattern, text, re.MULTILINE)
        found[name] = float(m.group(1)) if m else None
    return found


def gpu() -> dict[str, str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    ).stdout.strip()
    util, mem = (out.split(",") + ["", ""])[:2]
    return {"gpu_util_pct": util.strip(), "gpu_mem_mib": mem.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--every", type=float, default=2.0)
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["t", "container", "cpu_pct", "mem", *GAUGES, "gpu_util_pct", "gpu_mem_mib"]
    t0 = time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while time.time() - t0 < args.seconds:
            t = round(time.time() - t0, 1)
            shared = {**vllm_gauges(), **gpu()}
            for row in docker_stats():
                writer.writerow(
                    {
                        "t": t,
                        "container": row["Name"],
                        "cpu_pct": row["CPUPerc"],
                        "mem": row["MemUsage"],
                        **shared,
                    }
                )
            f.flush()
            time.sleep(args.every)


if __name__ == "__main__":
    main()
