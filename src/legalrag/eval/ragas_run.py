"""RAGAS scoring of saved predictions.

Faithfulness and answer relevancy (generation) are scored for answers that were not refused;
context precision and recall (retrieval) for every answerable question. The judge should be from
a different model family than the generator to avoid self-preference bias.
"""

from __future__ import annotations

import asyncio
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from legalrag.eval.metrics import Prediction
from legalrag.eval.pacing import Pacer
from legalrag.logging_conf import get_logger

log = get_logger(__name__)


class Metric(Protocol):
    async def ascore(self, **kwargs: Any) -> Any: ...


GENERATION = ("faithfulness", "answer_relevancy")
RETRIEVAL = ("context_precision", "context_recall")


def _value(result: Any) -> float | None:
    v = getattr(result, "value", result)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _judge_calls(name: str, p: Prediction) -> int:
    """Provider calls one metric makes, for pacing."""
    return {"faithfulness": 2, "context_precision": max(len(p.context_texts), 1)}.get(name, 1)


async def _score_one(p: Prediction, metrics: Mapping[str, Metric], pacer: Pacer) -> dict[str, Any]:
    row: dict[str, Any] = {"id": p.id, "lang": p.lang}
    if not p.answerable or p.error:
        return row
    calls = {
        "context_precision": dict(
            user_input=p.question, reference=p.reference, retrieved_contexts=p.context_texts
        ),
        "context_recall": dict(
            user_input=p.question, reference=p.reference, retrieved_contexts=p.context_texts
        ),
    }
    if not p.refused:
        calls["faithfulness"] = dict(
            user_input=p.question, response=p.answer, retrieved_contexts=p.context_texts
        )
        calls["answer_relevancy"] = dict(user_input=p.question, response=p.answer)
    for name, kwargs in calls.items():
        if name in metrics and p.context_texts:
            await pacer.wait(_judge_calls(name, p))
            try:
                row[name] = _value(await metrics[name].ascore(**kwargs))
            except Exception as exc:  # noqa: BLE001 - one failed metric must not lose the row
                row.setdefault("errors", []).append(f"{name}: {type(exc).__name__}")
                log.warning("ragas_metric_failed", id=p.id, metric=name, error=str(exc)[:200])
    return row


async def score_predictions(
    preds: Sequence[Prediction],
    metrics: Mapping[str, Metric],
    concurrency: int = 4,
    requests_per_minute: float | None = None,
) -> list[dict[str, Any]]:
    gate, pacer = asyncio.Semaphore(concurrency), Pacer(requests_per_minute)

    async def one(p: Prediction) -> dict[str, Any]:
        async with gate:
            return await _score_one(p, metrics, pacer)

    return list(await asyncio.gather(*(one(p) for p in preds)))


def summarize_ragas(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Mean of each metric over the rows that have it, overall and per language."""
    out: dict[str, dict[str, float | None]] = {}
    for group in ("all", "ar", "en"):
        subset = [r for r in rows if group == "all" or r["lang"] == group]
        if subset:
            out[group] = {
                m: (
                    statistics.fmean(vals)
                    if (vals := [r[m] for r in subset if r.get(m) is not None])
                    else None
                )
                for m in GENERATION + RETRIEVAL
            }
            out[group]["ragas_errors"] = sum(len(r.get("errors", [])) for r in subset)
    return out


def make_metrics(
    base_url: str,
    api_key: str,
    judge_model: str,
    embedding_model: str,
    names: Sequence[str] = GENERATION + RETRIEVAL,
    reasoning_effort: str | None = None,
) -> dict:
    """RAGAS 0.4 metrics judged through an OpenAI-compatible endpoint.

    ``names`` selects a subset; context precision costs one judge call per article.
    ``reasoning_effort="none"`` keeps thinking models from spending ``max_tokens`` on reasoning."""
    import os

    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=180, max_retries=5)
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    llm = llm_factory(judge_model, client=client, max_tokens=4096, **extra)  # long Arabic claims
    builders = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(
            llm=llm, embeddings=embedding_factory("openai", model=embedding_model, client=client)
        ),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
    }
    return {n: builders[n]() for n in names}
