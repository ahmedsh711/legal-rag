"""Token bucket in Redis (fakeredis runs the real Lua script)."""

from __future__ import annotations

import asyncio

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from legalrag.ratelimit import TokenBucket, client_key

fakeredis = pytest.importorskip("fakeredis")


@pytest.fixture
def redis():
    return fakeredis.FakeAsyncRedis()


async def test_a_burst_passes_then_a_429_with_an_honest_retry_after(redis):
    bucket = TokenBucket(redis, capacity=2, per_minute=1)  # one new token every 60 s
    first, second, third = [await bucket.take("u1") for _ in range(3)]
    assert first.allowed and second.allowed and second.remaining == 0
    assert not third.allowed and 55 <= third.retry_after_s <= 60  # when a token really exists


async def test_each_client_has_its_own_bucket(redis):
    bucket = TokenBucket(redis, capacity=1, per_minute=1)
    assert (await bucket.take("a")).allowed and (await bucket.take("b")).allowed
    assert not (await bucket.take("a")).allowed


async def test_tokens_come_back_over_time(redis):
    bucket = TokenBucket(redis, capacity=1, per_minute=600_000)  # ten tokens per millisecond
    assert (await bucket.take("u")).allowed
    await asyncio.sleep(0.01)
    assert (await bucket.take("u")).allowed


class DownRedis:
    def register_script(self, script):
        async def call(keys, args):
            raise RedisConnectionError("connection refused")

        return call


async def test_redis_outage_fails_open_and_says_so():
    verdict = await TokenBucket(DownRedis(), capacity=1, per_minute=1).take("u")
    assert verdict.allowed and verdict.degraded  # serving without a limit beats refusing all


def test_client_key_never_stores_an_api_key_in_clear():
    key = client_key("my-secret-key", "10.0.0.7")
    assert key.startswith("key:") and "my-secret-key" not in key
    assert client_key(None, "10.0.0.7") == "ip:10.0.0.7"
    assert client_key(None, None) == "ip:unknown"
