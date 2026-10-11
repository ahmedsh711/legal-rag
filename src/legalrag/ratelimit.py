"""Per-client rate limiting with a token bucket shared by all replicas through Redis."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from redis.exceptions import RedisError

from legalrag.logging_conf import get_logger

log = get_logger(__name__)

# KEYS[1] = bucket; ARGV[1] = capacity, ARGV[2] = refill in tokens per millisecond.
# Runs atomically in Redis and uses Redis TIME, so replicas with skewed clocks agree.
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
    degraded: bool = False  # Redis unreachable, allowed without checking


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
            # fail open: a dead Redis must not take the API down with it
            log.warning("ratelimit_unavailable", error=type(exc).__name__)
            return Verdict(allowed=True, remaining=-1, retry_after_s=0, degraded=True)
        return Verdict(bool(allowed), int(remaining), math.ceil(int(retry_ms) / 1000))

    async def aclose(self) -> None:
        await self.redis.aclose()


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def known_key_hashes(api_keys: str) -> frozenset[str]:
    """Hashes of the issued API keys (``API_KEYS``, comma-separated)."""
    return frozenset(_hash(k.strip()) for k in api_keys.split(",") if k.strip())


def client_key(api_key: str | None, ip: str | None, known: frozenset[str]) -> str:
    """Bucket key: the hashed API key if it was issued, otherwise the client IP.

    Unknown keys are ignored, otherwise a client could send a fresh key per request."""
    # TODO: behind a reverse proxy, take the IP from X-Forwarded-For set by that proxy
    if api_key and (hashed := _hash(api_key)) in known:
        return f"key:{hashed}"
    return f"ip:{ip or 'unknown'}"


def build_limiter(redis_url: str, per_minute: float, burst: int) -> TokenBucket | None:
    if per_minute <= 0:
        return None
    from redis.asyncio import Redis

    # a take costs ~1-2 ms; 0.2 s caps what a dead Redis adds to each request
    client = Redis.from_url(redis_url, socket_timeout=0.2, socket_connect_timeout=0.2)
    return TokenBucket(client, capacity=burst, per_minute=per_minute)
