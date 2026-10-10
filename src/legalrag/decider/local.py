"""Local decider: a cross-encoder reranker (bge-reranker-v2-m3) on CPU, no API key, no per-call
cost. A cross-encoder reads the question and the article together and outputs one relevance logit;
``sigmoid(logit)`` is the relevance, and the best article's relevance doubles as the gate score.

This is the standard alternative to Jev that the ablation compares against.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Callable, Sequence

from legalrag.decider.base import Decider, DeciderUnavailableError, Decision, passage_text
from legalrag.generation import detect_language
from legalrag.retrieval import Chunk

ScorePairs = Callable[[str, list[str]], list[float]]  # question, passages -> logits


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class LocalDecider(Decider):
    name = "local"

    def __init__(self, score_pairs: ScorePairs):
        self.score_pairs = score_pairs

    async def decide(self, question: str, chunks: Sequence[Chunk]) -> Decision:
        t0 = time.perf_counter()
        lang = detect_language(question)  # multilingual model: compare in the question's language
        texts = [passage_text(c, lang) for c in chunks]
        try:
            logits = await asyncio.to_thread(self.score_pairs, question, texts)  # CPU work
            relevance = {c.article_number: _sigmoid(x)
                         for c, x in zip(chunks, logits, strict=True)}  # fmt: skip
        except Exception as exc:  # torch/tokenizer failure: degrade like a Jev outage, not a 500
            raise DeciderUnavailableError(f"local: {type(exc).__name__}: {exc}") from exc
        return Decision(
            relevance=relevance,
            answerable=max(relevance.values(), default=0.0),
            decider=self.name,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    @classmethod
    def from_pretrained(
        cls, model_name: str, revision: str | None = None, device: str = "cpu"
    ) -> LocalDecider:
        import torch  # heavy imports kept local
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
        model = model.to(device).eval()
        # same reason as the embedder: the fast tokenizer is not thread-safe
        lock = threading.Lock()

        @torch.inference_mode()
        def score_pairs(question: str, texts: list[str]) -> list[float]:
            with lock:
                enc = tokenizer(
                    [[question, t] for t in texts],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                return model(**enc).logits.view(-1).float().cpu().tolist()

        return cls(score_pairs)
