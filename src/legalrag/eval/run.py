"""One experiment = the real pipeline over the golden set, scored and (optionally) logged.

    uv run python -m legalrag.eval.run --name baseline                       # deterministic metrics
    uv run python -m legalrag.eval.run --name jev --decider jev --ragas --mlflow

Needs Qdrant + the index + an LLM key (and the MLflow server for ``--mlflow``). Writes, under
``reports/eval/<name>/``: ``predictions.jsonl`` (answer, citations, context, tokens, latency per
question), ``metrics.json`` (overall + per language) and, with ``--ragas``, ``ragas.jsonl``.
RAGAS reads the saved predictions, so an expensive pipeline run is never repeated to add a metric.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from legalrag.eval.golden import GoldenItem, load_golden
from legalrag.eval.metrics import Prediction, gate_sweep, summarize
from legalrag.eval.pacing import Pacer
from legalrag.generation import PROMPT_VERSION, format_article
from legalrag.logging_conf import configure_logging, get_logger
from legalrag.settings import Settings

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
        answerable_score=ans.decision.answerable if ans.decision else None,
        gated=ans.gated,
        decider_cost_usd=ans.decision.cost_usd if ans.decision else 0.0,
        decider_model=ans.decision.model if ans.decision else "",
    )


async def predict_context(pipeline: Any, item: GoldenItem) -> Prediction:
    """Retrieval + decider only: no LLM call, no tokens (refused = the gate said no)."""
    t0 = time.perf_counter()
    ctx = await pipeline.select_context(item.question)
    return Prediction(
        id=item.id, lang=item.lang, category=item.category, question=item.question,
        gold_articles=item.gold_articles, reference=item.reference, answer="",
        refused=ctx.gated, gated=ctx.gated, cited=[],
        context_articles=[c.article_number for c in ctx.chunks],
        context_texts=[format_article(c) for c in ctx.chunks],
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        answerable_score=ctx.decision.answerable if ctx.decision else None,
        decider_cost_usd=ctx.decision.cost_usd if ctx.decision else 0.0,
        decider_model=ctx.decision.model if ctx.decision else "",
    )  # fmt: skip


async def run_golden(
    pipeline: Any,
    items: Sequence[GoldenItem],
    concurrency: int = 4,
    generate: bool = True,
    requests_per_minute: float | None = None,
) -> list[Prediction]:
    """All items, a few at a time and paced under the LLM's per-minute limit, in input order."""
    gate, pacer = asyncio.Semaphore(concurrency), Pacer(requests_per_minute if generate else None)
    step = predict if generate else predict_context

    async def one(item: GoldenItem) -> Prediction:
        async with gate:
            await pacer.wait()
            try:
                return await step(pipeline, item)
            except Exception as exc:  # noqa: BLE001 - one quota/timeout must not lose the run
                log.warning("eval_item_failed", id=item.id, error=str(exc)[:200])
                return Prediction(
                    id=item.id, lang=item.lang, category=item.category, question=item.question,
                    gold_articles=item.gold_articles, reference=item.reference, answer="",
                    refused=False, cited=[], context_articles=[],
                    error=f"{type(exc).__name__}: {exc}"[:300],
                )  # fmt: skip

    return list(await asyncio.gather(*(one(i) for i in items)))


# thresholds replayed offline; reported as correct_refusal - false_refusal (higher = better gate)
GATE_GRID = [round(0.05 * i, 2) for i in range(20)]

RETRIEVAL_KEYS = ("n", "errors", "hit_at_1", "hit_at_5", "mrr", "false_refusal_rate",
                  "correct_refusal_rate", "latency_p50_ms", "latency_p95_ms", "decider_cost_usd")  # fmt: skip


def summarize_retrieval(preds: Sequence[Prediction]) -> dict[str, dict[str, Any]]:
    """Only the metrics that mean something without an answer (refusals = the gate alone)."""
    return {g: {k: m[k] for k in RETRIEVAL_KEYS} for g, m in summarize(preds).items()}


def write_predictions(preds: Sequence[Prediction], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(p.model_dump(), ensure_ascii=False) + "\n" for p in preds)
    Path(path).write_text("".join(lines), encoding="utf-8", newline="\n")


def read_predictions(path: str | Path) -> list[Prediction]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Prediction(**json.loads(line)) for line in lines if line.strip()]


def rag_config(settings: Settings) -> dict[str, Any]:
    """Everything that changes an answer: logged as MLflow params, registered as the config."""
    return {
        "llm_model": settings.active_llm_model,
        "prompt_version": PROMPT_VERSION,
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_revision,
        "retrieval_mode": settings.retrieval_mode,
        "retrieve_top_n": settings.retrieve_top_n,
        "rerank_keep_top": settings.rerank_keep_top,
        "decider_backend": settings.decider_backend,
        "decider_model": {"jev": settings.jev_model, "local": settings.reranker_model}.get(
            settings.decider_backend, ""
        ),
        "gate_threshold": settings.gate_threshold,
        "llm_temperature": settings.llm_temperature,
    }


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one experiment over the golden set")
    p.add_argument("--name", required=True, help="run name; outputs go to reports/eval/<name>/")
    p.add_argument("--golden", default=None, help="default: settings.golden_set_path")
    p.add_argument("--decider", choices=["none", "jev", "local"], default=None)
    p.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default=None)
    p.add_argument("--context-size", type=int, default=None, help="articles shown to the LLM")
    p.add_argument("--top-n", type=int, default=None, help="articles retrieved")
    p.add_argument("--gate", type=float, default=None, help="answerability threshold")
    p.add_argument("--ragas", action="store_true", help="also score with the RAGAS judge")
    p.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    p.add_argument("--retrieval-only", action="store_true",
                   help="retrieval + decider only, no LLM (free): ranking and gate metrics")  # fmt: skip
    p.add_argument("--from-predictions", default=None,
                   help="score a saved predictions.jsonl instead of running the pipeline")  # fmt: skip
    p.add_argument("--concurrency", type=int, default=4)
    return p.parse_args(argv)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    pairs = {
        "decider_backend": args.decider,
        "retrieval_mode": args.mode,
        "rerank_keep_top": args.context_size,
        "retrieve_top_n": args.top_n,
        "gate_threshold": args.gate,
    }
    return {k: v for k, v in pairs.items() if v is not None}


def _write_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")  # fmt: skip


def _index_tags(settings: Settings) -> dict[str, str]:
    """Which index the run used (alias target + its metadata), without loading any model."""
    from qdrant_client import QdrantClient

    from legalrag.index.store import alias_target, read_metadata

    client = QdrantClient(url=settings.qdrant_url, timeout=10)
    collection = alias_target(client, settings.qdrant_collection_alias) or ""
    meta = (read_metadata(client, collection) or {}) if collection else {}
    return {"index_collection": collection, "articles_md5": str(meta.get("articles_md5", ""))}


def _md5(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()  # noqa: S324 - fingerprint only


def _git_dirty() -> str:
    import subprocess

    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, check=True, timeout=10)  # fmt: skip
        return str(bool(out.stdout.strip())).lower()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _lineage_tags(settings: Settings, args: argparse.Namespace, golden: Path,
                  preds: Sequence[Prediction]) -> dict[str, str]:  # fmt: skip
    """Everything needed to reproduce the run besides the params (git SHA is added by track)."""
    resolved = next((p.decider_model for p in preds if p.decider_model), "")
    return {
        "eval_mode": "retrieval_only" if args.retrieval_only else "end_to_end",
        "golden_md5": _md5(golden),
        "uv_lock_md5": _md5("uv.lock"),
        "git_dirty": _git_dirty(),
        "llm_backend": settings.llm_backend,
        "decider_resolved_model": resolved,  # e.g. the dated Jev build behind "jev-1.13"
        "judge_model": settings.judge_model if args.ragas else "",
        "judge_reasoning_effort": str(settings.judge_reasoning_effort or ""),
        "ragas_metrics": settings.ragas_metrics if args.ragas else "",
    }


def _ragas(preds: list[Prediction], settings: Settings, out: Path, concurrency: int) -> dict:
    from legalrag.eval.ragas_run import make_metrics, score_predictions, summarize_ragas

    judge = make_metrics(settings.judge_base_url, settings.judge_api_key,
                         settings.judge_model, settings.judge_embedding_model,
                         names=[m.strip() for m in settings.ragas_metrics.split(",") if m.strip()],
                         reasoning_effort=settings.judge_reasoning_effort)  # fmt: skip
    rows = asyncio.run(score_predictions(preds, judge, concurrency, settings.judge_rpm))
    lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / "ragas.jsonl").write_text(lines, encoding="utf-8", newline="\n")
    return summarize_ragas(rows)


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    from legalrag.settings import get_settings

    settings = get_settings().model_copy(update=_overrides(args))
    configure_logging(settings.log_level)
    out = Path("reports/eval") / args.name
    out.mkdir(parents=True, exist_ok=True)
    golden_path = Path(args.golden or settings.golden_set_path)
    if args.from_predictions:
        preds = read_predictions(args.from_predictions)
    else:
        from legalrag.api.main import build_components

        pipeline = build_components(settings).pipeline
        preds = asyncio.run(run_golden(pipeline, load_golden(golden_path), args.concurrency,
                                       generate=not args.retrieval_only,
                                       requests_per_minute=settings.llm_rpm))  # fmt: skip
    if args.retrieval_only:  # no LLM ran: every refusal is the gate's (also true for old files)
        preds = [p.model_copy(update={"gated": p.refused}) for p in preds]
    write_predictions(preds, out / "predictions.jsonl")
    if args.retrieval_only:
        summary = summarize_retrieval(preds)
        summary["gate_sweep"] = {f"t{r['threshold']:.2f}.{k}": r[k]
                                 for r in gate_sweep(preds, GATE_GRID)
                                 for k in ("false_refusal_rate", "correct_refusal_rate")}  # fmt: skip
    else:
        summary = summarize(preds, settings.llm_price_in_per_m, settings.llm_price_out_per_m)
    artifacts = [out / "predictions.jsonl", out / "metrics.json"]
    if args.ragas:
        for group, means in _ragas(preds, settings, out, args.concurrency).items():
            summary.setdefault(group, {}).update(means)
        artifacts.append(out / "ragas.jsonl")
    _write_json(summary, out / "metrics.json")
    if args.mlflow:
        import mlflow

        from legalrag.eval.track import log_eval_run

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        tags = {**_index_tags(settings), **_lineage_tags(settings, args, golden_path, preds)}
        run_id = log_eval_run(settings.mlflow_experiment, args.name, rag_config(settings),
                              summary, tags, artifacts)  # fmt: skip
        log.info("mlflow_run", run_id=run_id)
    log.info("eval_done", name=args.name, items=len(preds),
             **{k: summary["all"].get(k) for k in ("hit_at_5", "mrr", "faithfulness")})  # fmt: skip


if __name__ == "__main__":
    main()
