"""CI quality gate: the current prompt and model on 10 frozen questions, judged by RAGAS.

    uv run python -m legalrag.eval.smoke           # generate, judge, exit 1 on failure
    uv run python -m legalrag.eval.smoke freeze    # rebuild data/golden/smoke.jsonl

``smoke.jsonl`` stores each question with the articles the production config retrieved, so the
gate needs no Qdrant or corpus. Defaults: mean faithfulness >= 0.75, no answer below 0.5, a gold
article cited in >= 80% of answers, at most one unmeasured item. With n = 10 it catches gross
breakage only; the full golden run (``eval/run.py``) measures small differences.
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
    """Replays the stored articles for each question instead of searching."""

    def __init__(self, contexts: Mapping[str, list[Chunk]]):
        self.contexts = contexts

    def retrieve(
        self, question: str, top_n: int | None = None, book: str | None = None
    ) -> list[Chunk]:
        if question not in self.contexts:  # e.g. a guard rewrote it (PII placeholder)
            raise KeyError(f"no frozen articles for {question!r}")
        return list(self.contexts[question])


def freeze(predictions: Path, articles: Path, golden: Path, pairs: int = 5) -> list[dict[str, Any]]:
    """The first ``pairs`` in-scope AR/EN pairs of a run, with the articles it showed the model."""
    from legalrag.eval.track import git_sha
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
    source = f"{predictions.parent.name} (frozen at {git_sha()})"
    return [
        {
            **it.model_dump(),
            "frozen_from": source,
            "context": [payload(by_number[n], "ar") for n in shown[it.id]],
        }
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
        if item.question in contexts:
            raise ValueError(f"duplicate smoke question {item.id}: contexts are keyed by question")
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
    min_item: float = 0.5,
) -> dict[str, Any]:
    judged = [r for r in ragas_rows if r.get("faithfulness") is not None]
    scores = [r["faithfulness"] for r in judged]
    below_floor = [r["id"] for r in judged if r["faithfulness"] < min_item]
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
    if below_floor:
        reasons.append(f"answers below {min_item} faithfulness: {', '.join(below_floor)}")
    if cited is None or cited < min_cited:
        reasons.append(f"cited_gold {cited} < {min_cited}")
    return {
        "passed": not reasons,
        "faithfulness": faith,
        "cited_gold": cited,
        "below_floor": below_floor,
        "n": len(preds),
        "unmeasured": unmeasured,
        "reasons": reasons,
    }


def _summary_markdown(v: Mapping[str, Any], model: str) -> str:
    status = "PASSED" if v["passed"] else "FAILED"
    lines = [
        f"### RAGAS smoke gate: {status}",
        "",
        f"Generator: `{model}`",
        "",
        "| metric | value |",
        "|---|---|",
        f"| faithfulness | {v['faithfulness']} |",
        f"| cited_gold | {v['cited_gold']} |",
        f"| unmeasured items | {v['unmeasured']} of {v['n']} |",
    ]
    return "\n".join(lines + [f"- {r}" for r in v["reasons"]]) + "\n"


def prompt_under_test(prompt_file: Path | None, prompt_version: str | None) -> Any:
    """The code prompt, or a candidate text from a file (judged before it gets a label)."""
    from legalrag.observability.prompts import CodePrompt, StaticPrompt

    if prompt_file is None:
        return CodePrompt()
    return StaticPrompt(
        Path(prompt_file).read_text(encoding="utf-8"), prompt_version or "candidate"
    )


async def _generate_and_judge(
    s: Any, items: list[GoldenItem], contexts: dict[str, list[Chunk]], prompts: Any
) -> tuple[list[Prediction], list[dict[str, Any]]]:
    from legalrag.eval.ragas_run import make_metrics, score_predictions
    from legalrag.generation import Generator, make_client
    from legalrag.pipeline import RagPipeline

    client = make_client(s.active_llm_base_url, s.active_llm_api_key, s.llm_timeout_s)
    generator = Generator(client, s.active_llm_model, s.llm_max_tokens, s.llm_temperature)
    pipeline = RagPipeline(FrozenRetriever(contexts), generator, context_size=5, prompts=prompts)
    try:
        preds = await smoke_predictions(pipeline, items, s.llm_rpm)
    finally:
        await client.close()
    judge = make_metrics(
        s.judge_base_url,
        s.judge_api_key,
        s.judge_model,
        s.judge_embedding_model,
        names=["faithfulness"],
        reasoning_effort=s.judge_reasoning_effort,
    )
    return preds, await score_predictions(preds, judge, 2, s.judge_rpm)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from legalrag.settings import get_settings

    s = get_settings()
    from legalrag.observability.prompts import prompt_sha256

    items, contexts = load_smoke(Path(args.smoke))
    prompts = prompt_under_test(args.prompt_file, args.prompt_version)
    preds, rows = asyncio.run(_generate_and_judge(s, items, contexts, prompts))
    v = verdict(preds, rows, args.min_faithfulness, args.min_cited, args.max_errors, args.min_item)
    served = prompts.get()  # promotion checks the hash of this exact text
    v.update(prompt_version=served.version, prompt_sha256=prompt_sha256(served.text))
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
    parser.add_argument("--max-errors", type=int, default=1)
    parser.add_argument("--min-item", type=float, default=0.5)
    parser.add_argument("--out", default="reports/eval/smoke")
    parser.add_argument("--from-run", default="reports/eval/e2e-dense-jev/predictions.jsonl")
    parser.add_argument("--prompt-file", type=Path, help="judge this candidate system prompt")
    parser.add_argument("--prompt-version", help="its version name, e.g. v4")
    args = parser.parse_args(argv)
    configure_logging("INFO")
    if args.command == "freeze":
        rows = freeze(
            Path(args.from_run),
            Path("data/processed/articles.json"),
            Path("data/golden/golden_set.jsonl"),
        )
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        Path(args.smoke).write_text(text, encoding="utf-8", newline="\n")
        log.info("smoke_frozen", items=len(rows), path=args.smoke)
        return
    v = _run(args)
    log.info("smoke_gate", **v)
    sys.exit(0 if v["passed"] else 1)


if __name__ == "__main__":
    main()
