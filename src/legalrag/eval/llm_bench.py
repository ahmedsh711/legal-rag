"""Serving benchmark for the generation backend: TTFT, decode speed and throughput under load.

    uv run python -m legalrag.eval.llm_bench --name vllm-awq --levels 1 4 8 16 --vram

Every request is a real RAG prompt (the 10 smoke questions with their five articles, built by
``build_messages``) sent through ``Generator.stream``, the same code path as ``/ask?stream=true``.
For each concurrency level N, ``--requests`` requests run with exactly N in flight at a time.

- TTFT: time to first token, what the user feels as "it started answering".
- decode tok/s: tokens after the first one / time after the first one, per request.
- throughput tok/s: all completion tokens / wall time of the level, the server's capacity.
Writes ``reports/eval/llm-bench/<name>.json``. Warm-up requests run first and are not counted
(the first request pays for CUDA graph capture / connection setup).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legalrag.eval.metrics import _percentile
from legalrag.generation import Generator, build_messages
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)

Messages = list[dict[str, str]]


@dataclass(frozen=True)
class Sample:
    ttft_ms: float
    total_ms: float
    completion_tokens: int
    error: str = ""


async def measure(generator: Generator, messages: Messages) -> Sample:
    t0 = time.perf_counter()
    first = None
    try:
        stream = generator.stream(messages)
        async for _ in stream.tokens():
            if first is None:
                first = time.perf_counter()
    except Exception as exc:  # noqa: BLE001 - one overloaded request must not end the benchmark
        error = f"{type(exc).__name__}: {exc}"[:200]
        log.warning("bench_request_failed", error=error)  # a code bug shows up here, not as nulls
        return Sample(0.0, 0.0, 0, error)
    end = time.perf_counter()
    first = first or end
    return Sample((first - t0) * 1000, (end - t0) * 1000, stream.completion_tokens)


def _p(values: Sequence[float], q: float) -> float | None:
    return round(_percentile(values, q), 1) if values else None


def summarize(samples: Sequence[Sample], wall_s: float, concurrency: int) -> dict[str, Any]:
    ok = [s for s in samples if not s.error]
    decode = [  # the first token is TTFT; decode speed is the rest over the time after it
        (s.completion_tokens - 1) / ((s.total_ms - s.ttft_ms) / 1000)
        for s in ok
        if s.completion_tokens > 1 and s.total_ms > s.ttft_ms
    ]
    tokens = sum(s.completion_tokens for s in ok)
    return {
        "concurrency": concurrency,
        "requests": len(samples),
        "errors": len(samples) - len(ok),
        "ttft_p50_ms": _p([s.ttft_ms for s in ok], 50),
        "ttft_p95_ms": _p([s.ttft_ms for s in ok], 95),
        "latency_p50_ms": _p([s.total_ms for s in ok], 50),
        "latency_p95_ms": _p([s.total_ms for s in ok], 95),
        "decode_tok_s_p50": _p(decode, 50),
        "throughput_tok_s": round(tokens / wall_s, 1) if wall_s else None,
        "requests_per_s": round(len(ok) / wall_s, 2) if wall_s else None,
        "completion_tokens_p50": _p([s.completion_tokens for s in ok], 50),
        "first_error": next((s.error for s in samples if s.error), ""),
    }


def require_answers(level: dict[str, Any], what: str) -> None:
    """Stop when nothing was answered: a report of nulls from a dead server looks like data."""
    if level["requests"] and level["errors"] == level["requests"]:
        raise SystemExit(f"{what}: all {level['requests']} requests failed "
                         f"(first error: {level.get('first_error', '?')})")  # fmt: skip


async def run_level(
    generator: Generator, prompts: Sequence[Messages], concurrency: int, requests: int
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    gate = asyncio.Semaphore(concurrency)

    async def one(i: int) -> Sample:
        async with gate:
            return await measure(generator, prompts[i % len(prompts)])

    t0 = time.perf_counter()
    samples = await asyncio.gather(*(one(i) for i in range(requests)))
    return summarize(samples, time.perf_counter() - t0, concurrency)


def gpu_memory_mib() -> dict[str, int] | None:
    """Whole-GPU memory from nvidia-smi (includes the desktop's share on Windows)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout  # fmt: skip
        used, total = (int(x) for x in out.strip().splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):  # no GPU, "[N/A]", empty
        return None
    return {"used_mib": used, "total_mib": total}


def smoke_prompts() -> list[Messages]:
    from legalrag.eval.smoke import SMOKE_PATH, load_smoke

    items, contexts = load_smoke(SMOKE_PATH)
    return [build_messages(it.question, contexts[it.question]) for it in items]


async def _bench(args: argparse.Namespace) -> dict[str, Any]:
    from legalrag.generation import make_client
    from legalrag.settings import get_settings

    s = get_settings()
    client = make_client(s.active_llm_base_url, s.active_llm_api_key, s.llm_timeout_s)
    generator = Generator(client, s.active_llm_model, s.llm_max_tokens, s.llm_temperature)
    prompts = smoke_prompts()
    result: dict[str, Any] = {"backend": s.llm_backend, "model": s.active_llm_model,
                              "max_tokens": s.llm_max_tokens, "levels": []}  # fmt: skip
    try:
        warmup = await run_level(generator, prompts, concurrency=1, requests=args.warmup)
        require_answers(warmup, "warm-up")
        for n in args.levels:
            level = await run_level(generator, prompts, n, max(args.requests, n))
            require_answers(level, f"concurrency {n}")
            if args.vram:
                level["gpu"] = gpu_memory_mib()
            log.info("bench_level", **{k: v for k, v in level.items() if k != "gpu"})
            result["levels"].append(level)
    finally:
        await client.close()
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TTFT / tokens-per-second benchmark")
    parser.add_argument("--name", required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--requests", type=int, default=20, help="requests per level (>= N)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--vram", action="store_true", help="record nvidia-smi memory per level")
    args = parser.parse_args(argv)
    configure_logging("INFO")
    result = asyncio.run(_bench(args))
    if args.vram:  # vLLM reserves its share of VRAM up front: this is not "idle" memory
        result["gpu_after_run"] = gpu_memory_mib()
    out = Path("reports/eval/llm-bench")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8",
                                           newline="\n")  # fmt: skip


if __name__ == "__main__":
    main()
