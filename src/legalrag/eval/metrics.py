"""Deterministic metrics over pipeline predictions: no judge model, so they are free, fast and
exactly reproducible. They answer "did we find and cite the right article, refuse when we should,
answer in the right language, and how fast / how expensive?". RAGAS (an LLM judge) adds the
semantic metrics (faithfulness, relevancy) on top.

Every metric is reported overall and per language ("all", "ar", "en").
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from legalrag.generation import detect_language
from legalrag.retrieval import article_numbers_in

UNANSWERABLE = {"off_topic", "injection"}


class Prediction(BaseModel):
    """What the pipeline did for one golden item (one line of predictions.jsonl)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    lang: str
    category: str
    question: str
    gold_articles: list[int]
    answer: str
    refused: bool
    cited: list[int]  # articles the answer refers to (sources)
    context_articles: list[int]  # what the model was shown, in retrieval order
    invalid_citations: list[int] = []
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_texts: list[str] = []  # for RAGAS
    reference: str = ""
    answerable_score: float | None = None  # the decider's gate score, when one ran
    decider_cost_usd: float = 0.0

    @property
    def answerable(self) -> bool:
        return self.category not in UNANSWERABLE


def _rank(p: Prediction) -> int | None:
    """1-based position of the first gold article in the context, None if missing."""
    for i, n in enumerate(p.context_articles, start=1):
        if n in p.gold_articles:
            return i
    return None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _rate(preds: Sequence[Prediction], ok: Callable[[Prediction], bool]) -> float | None:
    return _mean([1.0 if ok(p) else 0.0 for p in preds])


def _block(preds: Sequence[Prediction], price_in: float, price_out: float) -> dict[str, Any]:
    answerable = [p for p in preds if p.answerable]
    unanswerable = [p for p in preds if not p.answerable]
    ranks = [_rank(p) for p in answerable]
    cited = [p for p in answerable if p.cited]
    latencies = [p.latency_ms for p in preds]
    return {
        "n": len(preds),
        "hit_at_1": _mean([1.0 if r == 1 else 0.0 for r in ranks]),
        "hit_at_5": _mean([1.0 if r is not None and r <= 5 else 0.0 for r in ranks]),
        "mrr": _mean([1 / r if r else 0.0 for r in ranks]),
        "citation_recall": _mean(
            [len(set(p.cited) & set(p.gold_articles)) / len(p.gold_articles) for p in answerable]
        ),
        "citation_precision": _mean(
            [len(set(p.cited) & set(p.gold_articles)) / len(p.cited) for p in cited]
        ),
        "false_refusal_rate": _rate(answerable, lambda p: p.refused),
        "correct_refusal_rate": _rate(unanswerable, lambda p: p.refused),
        "invalid_citation_rate": _rate(preds, lambda p: bool(p.invalid_citations)),
        "language_match": _rate(preds, lambda p: detect_language(p.answer) == p.lang),
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "latency_p95_ms": round(_percentile(latencies, 95), 1) if latencies else None,
        "prompt_tokens_mean": _mean([p.prompt_tokens for p in preds]),
        "completion_tokens_mean": _mean([p.completion_tokens for p in preds]),
        "decider_cost_usd": round(sum(p.decider_cost_usd for p in preds), 6),
        "cost_usd": round(  # LLM tokens + decider calls
            sum(p.prompt_tokens * price_in + p.completion_tokens * price_out for p in preds) / 1e6
            + sum(p.decider_cost_usd for p in preds),
            6,
        ),
    }


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def gate_sweep(preds: Sequence[Prediction], thresholds: Sequence[float]) -> list[dict[str, float]]:
    """Replay the gate offline: an item counts as refused if the run refused it or its decider
    score is below the threshold (questions naming an article are never gated, as in the
    pipeline). One row per threshold: the trade-off between the two refusal rates."""
    answerable = [p for p in preds if p.answerable]
    unanswerable = [p for p in preds if not p.answerable]
    rows = []
    for t in thresholds:

        def refused(p: Prediction, t: float = t) -> bool:
            gated = p.answerable_score is not None and p.answerable_score < t
            return p.refused or (gated and not article_numbers_in(p.question))

        rows.append({"threshold": t,
                     "false_refusal_rate": _rate(answerable, refused),
                     "correct_refusal_rate": _rate(unanswerable, refused)})  # fmt: skip
    return rows


def summarize(
    preds: Sequence[Prediction], price_in_per_m: float = 0.0, price_out_per_m: float = 0.0
) -> dict[str, dict[str, Any]]:
    """Metrics overall and per language. Prices are USD per million tokens."""
    out = {"all": _block(preds, price_in_per_m, price_out_per_m)}
    for lang in ("ar", "en"):
        subset = [p for p in preds if p.lang == lang]
        if subset:
            out[lang] = _block(subset, price_in_per_m, price_out_per_m)
    return out
