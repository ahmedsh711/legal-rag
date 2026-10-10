"""Dynamic micro-batching of query embeddings: questions that queue up share one forward pass."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from legalrag.index.batcher import QueryBatcher
from tests.conftest import FakeEmbedder


class SlowEmbedder(FakeEmbedder):
    """Like the real model: one call at a time, and a call costs time whatever its size."""

    def __init__(self):
        self.batches: list[int] = []
        self._busy = threading.Lock()

    def encode(self, texts, batch_size: int = 16):
        with self._busy:
            self.batches.append(len(texts))
            time.sleep(0.05)
            return super().encode(texts, batch_size)


def test_questions_that_arrive_together_share_a_forward_pass():
    inner = SlowEmbedder()
    batcher = QueryBatcher(inner, max_batch=16)
    questions = [f"question number {i}" for i in range(12)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda q: batcher.encode([q])[0], questions))
    batcher.close()
    assert results == FakeEmbedder().encode(questions)  # each caller gets its own vector back
    assert sum(inner.batches) == 12 and len(inner.batches) < 12  # fewer model calls than questions


def test_a_lone_question_is_not_delayed():
    inner = SlowEmbedder()
    batcher = QueryBatcher(inner)
    t0 = time.perf_counter()
    batcher.encode(["one question"])
    assert time.perf_counter() - t0 < 0.05 + 0.03  # the model's own time plus a little
    batcher.close()


def test_a_model_error_reaches_every_caller_in_the_batch():
    class Broken(FakeEmbedder):
        def encode(self, texts, batch_size=16):
            raise RuntimeError("model crashed")

    batcher = QueryBatcher(Broken())
    with pytest.raises(RuntimeError, match="model crashed"):
        batcher.encode(["q"])
    batcher.close()


def test_many_texts_at_once_go_straight_to_the_model():
    inner = SlowEmbedder()
    batcher = QueryBatcher(inner)
    batcher.encode(["a", "b", "c"])  # e.g. building the index: already a batch
    assert inner.batches == [3]
    assert batcher.dim == inner.dim  # everything else is the embedder's own
    batcher.close()
