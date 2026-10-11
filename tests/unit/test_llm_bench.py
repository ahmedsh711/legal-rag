"""Serving benchmark: TTFT and tokens/s measured on the same streaming path the API uses."""

from __future__ import annotations

import asyncio

import pytest

from legalrag.eval.llm_bench import (
    Sample,
    gpu_memory_mib,
    measure,
    require_answers,
    run_level,
    summarize,
)
from legalrag.generation import Generator
from tests.fakes import FakeLLM


def test_summary_reports_percentiles_decode_speed_and_throughput():
    samples = [Sample(ttft_ms=100, total_ms=1100, completion_tokens=50) for _ in range(4)]
    samples.append(Sample(ttft_ms=0, total_ms=0, completion_tokens=0, error="Timeout"))
    s = summarize(samples, wall_s=2.0, concurrency=4)
    assert s["requests"] == 5 and s["errors"] == 1
    assert s["ttft_p50_ms"] == 100 and s["latency_p95_ms"] == 1100
    assert s["decode_tok_s_p50"] == 49.0  # 49 tokens in the 1.0 s after the first one
    assert s["throughput_tok_s"] == 100.0  # 200 tokens in 2 s of wall time


async def test_measure_times_the_first_token_and_counts_tokens():
    gen = Generator(FakeLLM("one two three"), "m", 100, 0.0)
    sample = await measure(gen, [{"role": "user", "content": "hi"}])
    assert sample.completion_tokens == 30  # from the usage chunk, like vLLM sends it
    assert 0 <= sample.ttft_ms <= sample.total_ms and not sample.error


class SlowLLM(FakeLLM):
    in_flight = peak = 0

    async def _create(self, **kwargs):
        SlowLLM.in_flight += 1
        SlowLLM.peak = max(SlowLLM.peak, SlowLLM.in_flight)
        await asyncio.sleep(0.01)
        SlowLLM.in_flight -= 1
        return await super()._create(**kwargs)


async def test_a_level_keeps_exactly_n_requests_in_flight():
    gen = Generator(SlowLLM("a b"), "m", 100, 0.0)
    result = await run_level(gen, [[{"role": "user", "content": "q"}]], concurrency=3, requests=9)
    assert result["requests"] == 9 and SlowLLM.peak == 3


async def test_a_failed_request_is_counted_not_fatal():
    class Broken(FakeLLM):
        async def _create(self, **kwargs):
            raise TimeoutError("vllm busy")

    result = await run_level(Generator(Broken("x"), "m", 1, 0.0), [[]], concurrency=1, requests=2)
    assert result["errors"] == 2 and result["ttft_p50_ms"] is None


@pytest.mark.parametrize("bad", [0, -1])
async def test_concurrency_must_be_positive(bad):
    with pytest.raises(ValueError):
        await run_level(Generator(FakeLLM("x"), "m", 1, 0.0), [[]], concurrency=bad, requests=1)


def test_a_level_where_every_request_failed_stops_the_benchmark():
    # a dead server must not produce a report full of nulls with exit code 0
    with pytest.raises(SystemExit, match="all 2 requests failed"):
        require_answers({"requests": 2, "errors": 2}, "warm-up")
    require_answers({"requests": 2, "errors": 1}, "level 4")  # partial failure: keep going


def test_unreadable_nvidia_smi_output_is_no_reading_not_a_crash(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="[N/A], [N/A]"))
    assert gpu_memory_mib() is None


async def test_repeated_prompts_are_made_unique_so_the_prefix_cache_cannot_flatter_ttft():
    llm = FakeLLM("a b")
    prompt = [{"role": "system", "content": "rules"}, {"role": "user", "content": "q?"}]
    await run_level(Generator(llm, "m", 10, 0.0), [prompt], concurrency=2, requests=4)
    users = [c["messages"][1]["content"] for c in llm.calls]
    assert len(set(users)) == 4 and all(u.endswith("q?") for u in users)
    assert {c["messages"][0]["content"] for c in llm.calls} == {"rules"}  # shared prefix stays
    assert prompt[1]["content"] == "q?"  # the caller's prompt is not changed
