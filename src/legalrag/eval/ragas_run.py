"""RAGAS: an LLM judge for the four classic RAG metrics, each blaming a different stage.

- faithfulness: is every claim in the answer supported by the retrieved articles? (generation)
- answer relevancy: does the answer address the question? (generation)
- context precision: are the relevant articles ranked high? (retrieval/rerank)
- context recall: do the retrieved articles cover the reference answer? (retrieval)

Faithfulness and relevancy are only scored for answers the system gave (a refusal has no claims);
precision and recall for every answerable question. The judge is a different model family from
the generator (self-preference bias), called through OpenRouter.
"""

from __future__ import annotations

import asyncio
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from legalrag.eval.metrics import Prediction
from legalrag.logging_conf import get_logger

log = get_logger(__name__)


class Metric(Protocol):
    async def ascore(self, **kwargs: Any) -> Any: ...


GENERATION = ("faithfulness", "answer_relevancy")
RETRIEVAL = ("context_precision", "context_recall")


def _value(result: Any) -> float | None:
    v = getattr(result, "value", result)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


async def _score_one(p: Prediction, metrics: Mapping[str, Metric]) -> dict[str, Any]:
    row: dict[str, Any] = {"id": p.id, "lang": p.lang}
    if not p.answerable:
        return row
    calls = {
        "context_precision": dict(user_input=p.question, reference=p.reference,
                                  retrieved_contexts=p.context_texts),
        "context_recall": dict(user_input=p.question, reference=p.reference,
                               retrieved_contexts=p.context_texts),
    }  # fmt: skip
    if not p.refused:
        calls["faithfulness"] = dict(
            user_input=p.question, response=p.answer, retrieved_contexts=p.context_texts
        )
        calls["answer_relevancy"] = dict(user_input=p.question, response=p.answer)
    for name, kwargs in calls.items():
        if name in metrics and p.context_texts:
            try:
                row[name] = _value(await metrics[name].ascore(**kwargs))
            except Exception as exc:  # noqa: BLE001 - rate limit / credits / bad JSON: keep the rest
                row.setdefault("errors", []).append(f"{name}: {type(exc).__name__}")
                log.warning("ragas_metric_failed", id=p.id, metric=name, error=str(exc)[:200])
    return row


async def score_predictions(
    preds: Sequence[Prediction], metrics: Mapping[str, Metric], concurrency: int = 4
) -> list[dict[str, Any]]:
    gate = asyncio.Semaphore(concurrency)

    async def one(p: Prediction) -> dict[str, Any]:
        async with gate:
            return await _score_one(p, metrics)

    return list(await asyncio.gather(*(one(p) for p in preds)))


def summarize_ragas(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Mean of each metric over the rows that have it, overall and per language."""
    out: dict[str, dict[str, float | None]] = {}
    for group in ("all", "ar", "en"):
        subset = [r for r in rows if group == "all" or r["lang"] == group]
        if subset:
            out[group] = {
                m: (statistics.fmean(vals) if (vals := [r[m] for r in subset
                                                         if r.get(m) is not None]) else None)
                for m in GENERATION + RETRIEVAL
            }  # fmt: skip
            out[group]["ragas_errors"] = sum(len(r.get("errors", [])) for r in subset)
    return out


def make_metrics(base_url: str, api_key: str, judge_model: str, embedding_model: str) -> dict:
    """The real RAGAS 0.4 metrics, judged through an OpenAI-compatible endpoint (OpenRouter)."""
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

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=2)
    llm = llm_factory(judge_model, client=client, max_tokens=4096)  # long Arabic statements
    emb = embedding_factory("openai", model=embedding_model, client=client)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }
