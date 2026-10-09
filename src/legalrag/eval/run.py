"""Run the real pipeline over the golden set and score it.

    uv run python -m legalrag.eval.run                     # needs Qdrant + the index + an LLM key

Writes ``reports/eval/predictions.jsonl`` (one line per question: answer, citations, context,
tokens, latency) and ``reports/eval/metrics.json`` (``eval.metrics.summarize``, overall and per
language). Later steps (RAGAS, the judge, MLflow) read the predictions file instead of calling
the pipeline again, so an expensive run is never repeated just to add a metric.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from legalrag.eval.golden import GoldenItem, load_golden
from legalrag.eval.metrics import Prediction, summarize
from legalrag.generation import format_article
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)


async def predict(pipeline: Any, item: GoldenItem) -> Prediction:
    t0 = time.perf_counter()
    ans = await pipeline.ask(item.question)
    return Prediction(
        id=item.id,
        lang=item.lang,
        category=item.category,
        question=item.question,
        gold_articles=item.gold_articles,
        reference=item.reference,
        answer=ans.answer,
        refused=ans.refused,
        cited=[c.article_number for c in ans.sources],
        context_articles=[c.article_number for c in ans.context],
        context_texts=[format_article(c) for c in ans.context],
        invalid_citations=ans.invalid_citations,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        prompt_tokens=ans.usage.get("prompt_tokens", 0),
        completion_tokens=ans.usage.get("completion_tokens", 0),
    )


async def run_golden(
    pipeline: Any, items: Sequence[GoldenItem], concurrency: int = 4
) -> list[Prediction]:
    """All items, a few at a time (provider rate limits), results in input order."""
    gate = asyncio.Semaphore(concurrency)

    async def one(item: GoldenItem) -> Prediction:
        async with gate:
            return await predict(pipeline, item)

    return list(await asyncio.gather(*(one(i) for i in items)))


def write_predictions(preds: Sequence[Prediction], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(p.model_dump(), ensure_ascii=False) + "\n" for p in preds)
    Path(path).write_text("".join(lines), encoding="utf-8", newline="\n")


def read_predictions(path: str | Path) -> list[Prediction]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Prediction(**json.loads(line)) for line in lines if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline over the golden set")
    parser.add_argument("--golden", default=None, help="default: settings.golden_set_path")
    parser.add_argument("--out", default="reports/eval/predictions.jsonl")
    parser.add_argument("--metrics", default="reports/eval/metrics.json")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)

    from legalrag.api.main import build_components
    from legalrag.settings import get_settings

    settings = get_settings()
    configure_logging(settings.log_level)
    items = load_golden(args.golden or settings.golden_set_path)
    pipeline = build_components(settings).pipeline
    preds = asyncio.run(run_golden(pipeline, items, args.concurrency))
    write_predictions(preds, args.out)
    summary = summarize(preds, settings.llm_price_in_per_m, settings.llm_price_out_per_m)
    Path(args.metrics).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    log.info("eval_done", items=len(preds), **{k: summary["all"][k] for k in ("hit_at_5", "mrr")})


if __name__ == "__main__":
    main()
