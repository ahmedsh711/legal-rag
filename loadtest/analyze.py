"""Per-step table of a stepped load test: which stage slows down, and which resource is full.

    docker logs --since 30m legal-rag-api-1 2>&1 | grep '"event": "answered"' > reports/load/<name>-api.jsonl
    uv run python loadtest/analyze.py --name <name> --steps 5,10,20,40 --hold 120

Joins three sources by time: the API's own "answered" log lines (per-stage timings of every
answer), Locust's history CSV (when the test started) and sample_stats.py (CPU, vLLM queue, GPU).
Writes reports/load/<name>-steps.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from legalrag.eval.metrics import _percentile


def _p(values: list[float], q: float) -> str:
    return f"{_percentile(values, q):,.0f}" if values else "-"


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.0f}" if values else "-"


def load_answers(path: Path, start: float) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        t = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp()
        if t >= start:
            rows.append({**row, "t": t - start})
    return rows


def load_samples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def step_rows(answers: list[dict], samples: list[dict], steps: list[int], hold: int) -> list[str]:
    lines = ["| users | answers/s | retrieve p50 / p95 ms | generate p50 / p95 ms | total p50 / p95 ms"
             " | API CPU % mean / max | vLLM running max · waiting max | GPU util % mean |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for k, users in enumerate(steps):
        lo, hi = k * hold, (k + 1) * hold
        a = [r for r in answers if lo <= r["t"] < hi]
        s = [r for r in samples if lo <= float(r["t"]) < hi]
        api = [float(r["cpu_pct"].rstrip("%")) for r in s if r["container"].endswith("api-1")]
        running = [float(r["vllm_running"]) for r in s if r["vllm_running"]]
        waiting = [float(r["vllm_waiting"]) for r in s if r["vllm_waiting"]]
        gpu = [
            float(r["gpu_util_pct"])
            for r in s
            if r["gpu_util_pct"] and r["container"].endswith("api-1")
        ]
        col = {key: [r[key] for r in a if key in r] for key in ("retrieve", "generate", "total")}
        lines.append(
            f"| {users} | {len(a) / hold:.2f} | {_p(col['retrieve'], 50)} / {_p(col['retrieve'], 95)}"
            f" | {_p(col['generate'], 50)} / {_p(col['generate'], 95)}"
            f" | {_p(col['total'], 50)} / {_p(col['total'], 95)}"
            f" | {_mean(api)} / {max(api, default=0):.0f}"
            f" | {max(running, default=0):.0f} · {max(waiting, default=0):.0f} | {_mean(gpu)} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="per-step load-test table")
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", default="5,10,20,40")
    parser.add_argument("--hold", type=int, default=120)
    args = parser.parse_args()
    base = Path("reports/load")
    with (base / f"{args.name}_stats_history.csv").open(encoding="utf-8") as f:
        start = float(next(csv.DictReader(f))["Timestamp"])
    answers = load_answers(base / f"{args.name}-api.jsonl", start)
    samples = load_samples(base / f"{args.name}-stats.csv")
    steps = [int(n) for n in args.steps.split(",")]
    table = "\n".join(step_rows(answers, samples, steps, args.hold)) + "\n"
    (base / f"{args.name}-steps.md").write_text(table, encoding="utf-8", newline="\n")
    print(table)  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
