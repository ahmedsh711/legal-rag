"""Serving benchmark: TTFT and tokens/s measured on the same streaming path the API uses."""

from __future__ import annotations

import asyncio

import pytest

from legalrag.eval.llm_bench import Sample, measure, run_level, summarize
from legalrag.generation import Generator
from tests.fakes import FakeLLM


def test_summary_reports_percentiles_decode_speed_and_throughput():
    samples = [Sample(ttft_ms=100, total_ms=1100, completion_tokens=50) for _ in range(4)]
    samples.append(Sample(ttft_ms=0, total_ms=0, completion_tokens=0, error="Timeout"))
    s = summarize(samples, wall_s=2.0, concurrency=4)
    assert s["requests"] == 5 and s["errors"] == 1
    assert s["ttft_p50_ms"] == 100 and s["latency_p95_ms"] == 1100
    assert s["decode_tok_s_p50"] == 50.0  # 50 tokens in the 1.0 s after the first one
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
