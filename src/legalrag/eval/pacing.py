"""Stay under a provider's requests-per-minute limit instead of hammering it and retrying.

Free tiers count requests per minute (Gemini Flash-Lite: 15). Spacing request *starts* evenly is
the simplest way to never hit the limit; retries stay for the rare surprise.
"""

from __future__ import annotations

import asyncio
import time


class Pacer:
    def __init__(self, per_minute: float | None):
        self.interval = 60.0 / per_minute if per_minute else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self, requests: int = 1) -> None:
        """Return when the next step may start; it will make ``requests`` provider calls."""
        if not self.interval:
            return
        async with self._lock:  # reserve the slots; sleep outside the lock
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval * requests
        if start > now:
            await asyncio.sleep(start - now)
