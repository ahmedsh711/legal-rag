"""Dynamic micro-batching: questions that queue up while the embedder is busy share one pass."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Any

from legalrag.logging_conf import get_logger

log = get_logger(__name__)

_STOP = object()


class QueryBatcher:
    def __init__(self, embedder: Any, max_batch: int = 16):
        self.embedder, self.max_batch = embedder, max_batch
        self._queue: queue.Queue[Any] = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="query-batcher", daemon=True)
        self._worker.start()

    def __getattr__(self, name: str) -> Any:  # dim, model_name, revision come from the embedder
        return getattr(self.embedder, name)

    def encode(self, texts: list[str], batch_size: int = 16) -> list[Any]:
        if len(texts) != 1:  # already a batch (index build), embed directly
            return self.embedder.encode(texts, batch_size)
        future: Future[Any] = Future()
        self._queue.put((texts[0], future))
        return [future.result(timeout=60)]

    def _next_batch(self) -> list[tuple[str, Future[Any]]] | None:
        first = self._queue.get()  # block until there is work
        if first is _STOP:
            return None
        batch = [first]
        while len(batch) < self.max_batch:  # drain what queued meanwhile, without waiting
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                self._queue.put(_STOP)  # finish this batch, stop on the next round
                break
            batch.append(item)
        return batch

    def _run(self) -> None:
        while (batch := self._next_batch()) is not None:
            try:
                vectors = self.embedder.encode([text for text, _ in batch], self.max_batch)
            except Exception as exc:  # noqa: BLE001 - every waiting caller gets the error
                for _, future in batch:
                    future.set_exception(exc)
                continue
            for (_, future), vector in zip(batch, vectors, strict=True):
                future.set_result(vector)
            if len(batch) > 1:
                log.debug("query_batch", size=len(batch))

    def close(self) -> None:
        self._queue.put(_STOP)
        self._worker.join(timeout=5)
