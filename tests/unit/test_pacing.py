"""Client-side pacing for free tiers that count requests per minute."""

import asyncio
import time

from legalrag.eval.pacing import Pacer


async def test_pacer_spaces_starts_evenly():
    pacer = Pacer(per_minute=600)  # one start every 0.1 s
    t0 = time.perf_counter()
    await asyncio.gather(*(pacer.wait() for _ in range(4)))
    assert time.perf_counter() - t0 >= 0.29  # starts at 0, 0.1, 0.2, 0.3


async def test_a_step_that_makes_two_calls_reserves_two_slots():
    pacer = Pacer(per_minute=600)
    t0 = time.perf_counter()
    await pacer.wait(requests=2)  # starts now, next start in 0.2 s
    await pacer.wait()
    assert time.perf_counter() - t0 >= 0.19


async def test_no_limit_means_no_waiting():
    pacer = Pacer(per_minute=None)
    t0 = time.perf_counter()
    await asyncio.gather(*(pacer.wait() for _ in range(50)))
    assert time.perf_counter() - t0 < 0.05
