"""Per-client rate limit: a token bucket in Redis, answered with 429 + an honest Retry-After.

- Token bucket: a client may burst up to ``capacity`` requests, then gets ``per_minute`` new
  tokens per minute. Fair to people who ask three questions in a row, firm with a script.
- Redis + one Lua script: every API replica shares the same buckets, and "refill, take, save"
  runs atomically inside Redis (two requests of one client cannot both take the last token).
  The clock is Redis's own (``TIME``), so replicas with skewed clocks still agree.
- Honest Retry-After: the seconds until a token really exists, not a fixed guess, so a
  well-behaved client retries once instead of hammering.
- Fail open: if Redis is down the request is served without a limit and a warning is logged.
  A rate limiter that takes the whole API down with it does more harm than the traffic it stops.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from redis.exceptions import RedisError

from legalrag.logging_conf import get_logger

log = get_logger(__name__)

# KEYS[1] = bucket; ARGV[1] = capacity, ARGV[2] = refill in tokens per millisecond
_TAKE = """
local cap = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local saved = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(saved[1]) or cap
local ts = tonumber(saved[2]) or now
tokens = math.min(cap, tokens + (now - ts) * rate)
local retry_ms = 0
if tokens >= 1 then
  tokens = tokens - 1
else
  retry_ms = math.ceil((1 - tokens) / rate)
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('PEXPIRE', KEYS[1], math.ceil(cap / rate) + 1000)
local allowed = 0
if retry_ms == 0 then allowed = 1 end
return {allowed, math.floor(tokens), retry_ms}
"""


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    remaining: int  # whole tokens left after this request
    retry_after_s: int  # 0 when allowed
    degraded: bool = False  # Redis unreachable: allowed without checking


class TokenBucket:
    def __init__(self, redis: Any, capacity: int, per_minute: float, prefix: str = "ratelimit"):
        self.redis, self.capacity, self.prefix = redis, capacity, prefix
        self.per_ms = per_minute / 60_000
        self._take = redis.register_script(_TAKE)

    async def take(self, client: str) -> Verdict:
        try:
            allowed, remaining, retry_ms = await self._take(
                keys=[f"{self.prefix}:{client}"], args=[self.capacity, repr(self.per_ms)]
            )
        except RedisError as exc:
            log.warning("ratelimit_unavailable", error=type(exc).__name__)
            return Verdict(allowed=True, remaining=-1, retry_after_s=0, degraded=True)
        return Verdict(bool(allowed), int(remaining), math.ceil(int(retry_ms) / 1000))

    async def aclose(self) -> None:
        await self.redis.aclose()


def client_key(api_key: str | None, ip: str | None) -> str:
    """Who is asking: the API key if one is sent (hashed, never stored in clear), else the IP.
    # ponytail: behind a proxy (Phase 6 nginx) the IP must come from X-Forwarded-For."""
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return f"ip:{ip or 'unknown'}"


def build_limiter(redis_url: str, per_minute: float, burst: int) -> TokenBucket | None:
    """None when the limit is switched off (per_minute <= 0)."""
    if per_minute <= 0:
        return None
    from redis.asyncio import Redis

    # measured: a take costs p50 0.8 ms / p95 1.9 ms (host -> container). 0.2 s is 100x that,
    # and caps what a dead Redis adds to each question.
    # ponytail: a circuit breaker would skip even those 0.2 s while Redis is down
    client = Redis.from_url(redis_url, socket_timeout=0.2, socket_connect_timeout=0.2)
    return TokenBucket(client, capacity=burst, per_minute=per_minute)
