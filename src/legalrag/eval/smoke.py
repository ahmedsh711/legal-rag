"""CI quality gate: the current prompt + model on 10 frozen questions, judged by RAGAS.

    uv run python -m legalrag.eval.smoke                  # generate, judge, pass/fail (exit 1)
    uv run python -m legalrag.eval.smoke freeze           # rebuild data/golden/smoke.jsonl

The CI runner has no Qdrant, no bge-m3 and no articles.json (DVC), and this gate does not need
them: ``data/golden/smoke.jsonl`` holds 10 golden questions (5 pairs, AR + EN) with the exact
five articles the production config (dense + Jev) put in front of the model. Generation runs
through the real ``RagPipeline`` (guards, prompt, citation check) with a retriever that returns
those articles. So the gate guards the prompt and the generation model; retrieval is guarded by
exact metrics on the full golden set (``eval/run.py``).

Two thresholds, both must hold:
- faithfulness (RAGAS judge) >= 0.75, mean over the answered items;
- cited_gold: share of answers that cite a gold article >= 0.8 (exact, no judge).
It fails closed: if more than 2 items could not be generated or judged (quota, outage), the gate
fails. A gate that passes whenever it cannot measure is not a gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from legalrag.eval.golden import GoldenItem, load_golden
from legalrag.eval.metrics import Prediction
from legalrag.eval.run import run_golden
from legalrag.logging_conf import configure_logging, get_logger
from legalrag.retrieval import Chunk

log = get_logger(__name__)

SMOKE_PATH = Path("data/golden/smoke.jsonl")
GOLDEN_FIELDS = set(GoldenItem.model_fields)


class FrozenRetriever:
    """Returns the articles stored for a question: retrieval replayed, not recomputed."""

    def __init__(self, contexts: Mapping[str, list[Chunk]]):
        self.contexts = contexts

    def retrieve(self, question: str, top_n: int | None = None, book: str | None = None):
        return list(self.contexts[question])


def freeze(predictions: Path, articles: Path, golden: Path, pairs: int = 5) -> list[dict]:
    """Smoke rows from a production run: the first ``pairs`` in-scope pairs, AR + EN, each
    with the full payload of the articles that run showed the model."""
    from legalrag.index.store import payload
    from legalrag.ingest.validate import load_articles

    by_number = {a.article_number: a for a in load_articles(articles)}
    shown = {}
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            shown[row["id"]] = row["context_articles"]
    items = [it for it in load_golden(golden) if it.category == "in_scope" and it.id in shown]
    keep = sorted({it.pair for it in items})[:pairs]
    return [
        {**it.model_dump(), "context": [payload(by_number[n], "ar") for n in shown[it.id]]}
        for it in items
        if it.pair in keep
    ]


def load_smoke(path: Path) -> tuple[list[GoldenItem], dict[str, list[Chunk]]]:
    items, contexts = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        item = GoldenItem(**{k: v for k, v in row.items() if k in GOLDEN_FIELDS})
        items.append(item)
        contexts[item.question] = [Chunk.from_payload(p) for p in row["context"]]
    return items, contexts


async def smoke_predictions(
    pipeline: Any, items: Sequence[GoldenItem], rpm: float | None = None
) -> list[Prediction]:
    return await run_golden(pipeline, items, concurrency=2, requests_per_minute=rpm)


def verdict(
    preds: Sequence[Prediction],
    ragas_rows: Sequence[Mapping[str, Any]],
    min_faithfulness: float,
    min_cited: float,
    max_errors: int,
) -> dict[str, Any]:
    scores = [r["faithfulness"] for r in ragas_rows if r.get("faithfulness") is not None]
    ok = [p for p in preds if not p.error]
    unmeasured = len(preds) - len(scores)
    faith = round(sum(scores) / len(scores), 3) if scores else None
    cited = (
        round(sum(bool(set(p.cited) & set(p.gold_articles)) for p in ok) / len(ok), 3)
        if ok
        else None
    )
    reasons = []
    if unmeasured > max_errors:
        reasons.append(
            f"{unmeasured} items could not be measured (max {max_errors}): failing closed"
        )
    if faith is None or faith < min_faithfulness:
        reasons.append(f"faithfulness {faith} < {min_faithfulness}")
    if cited is None or cited < min_cited:
        reasons.append(f"cited_gold {cited} < {min_cited}")
    return {"passed": not reasons, "faithfulness": faith, "cited_gold": cited,
            "n": len(preds), "unmeasured": unmeasured, "reasons": reasons}  # fmt: skip


def _summary_markdown(v: Mapping[str, Any], model: str) -> str:
    status = "PASSED" if v["passed"] else "FAILED"
    lines = [f"### RAGAS smoke gate: {status}", "", f"Generator: `{model}`", "",
             "| metric | value |", "|---|---|",
             f"| faithfulness | {v['faithfulness']} |", f"| cited_gold | {v['cited_gold']} |",
             f"| unmeasured items | {v['unmeasured']} of {v['n']} |"]  # fmt: skip
    return "\n".join(lines + [f"- {r}" for r in v["reasons"]]) + "\n"


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from legalrag.eval.ragas_run import make_metrics, score_predictions
    from legalrag.generation import Generator, make_client
    from legalrag.pipeline import RagPipeline
    from legalrag.settings import get_settings

    s = get_settings()
    items, contexts = load_smoke(Path(args.smoke))
    generator = Generator(make_client(s.active_llm_base_url, s.active_llm_api_key, s.llm_timeout_s),
                          s.active_llm_model, s.llm_max_tokens, s.llm_temperature)  # fmt: skip
    pipeline = RagPipeline(FrozenRetriever(contexts), generator, context_size=5)
    preds = asyncio.run(smoke_predictions(pipeline, items, s.llm_rpm))
    judge = make_metrics(s.judge_base_url, s.judge_api_key, s.judge_model,
                         s.judge_embedding_model, names=["faithfulness"],
                         reasoning_effort=s.judge_reasoning_effort)  # fmt: skip
    rows = asyncio.run(score_predictions(preds, judge, 2, s.judge_rpm))
    v = verdict(preds, rows, args.min_faithfulness, args.min_cited, args.max_errors)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("predictions.jsonl", [p.model_dump() for p in preds]),
        ("ragas.jsonl", rows),
    ):
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    (out / "verdict.json").write_text(
        json.dumps(v, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):  # shown on the CI run page
        with open(summary, "a", encoding="utf-8") as f:
            f.write(_summary_markdown(v, s.active_llm_model))
    return v


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RAGAS smoke gate for CI")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "freeze"])
    parser.add_argument("--smoke", default=str(SMOKE_PATH))
    parser.add_argument("--min-faithfulness", type=float, default=0.75)
    parser.add_argument("--min-cited", type=float, default=0.8)
    parser.add_argument("--max-errors", type=int, default=2)
    parser.add_argument("--out", default="reports/eval/smoke")
    parser.add_argument("--from-run", default="reports/eval/e2e-dense-jev/predictions.jsonl")
    args = parser.parse_args(argv)
    configure_logging("INFO")
    if args.command == "freeze":
        rows = freeze(Path(args.from_run), Path("data/processed/articles.json"),
                      Path("data/golden/golden_set.jsonl"))  # fmt: skip
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        Path(args.smoke).write_text(text, encoding="utf-8", newline="\n")
        log.info("smoke_frozen", items=len(rows), path=args.smoke)
        return
    v = _run(args)
    log.info("smoke_gate", **v)
    sys.exit(0 if v["passed"] else 1)


if __name__ == "__main__":
    main()
