"""Measure the input guardrails: detection rate, false positives and latency.

    uv run python -m legalrag.eval.guardrails_eval [--mlflow]

Scores the attack set (``data/golden/guardrails.jsonl``), checks golden and held-out questions and
the corpus articles for false positives, and writes ``reports/eval/guardrails/summary.json``.
No model or network needed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from legalrag.eval.metrics import _percentile
from legalrag.guardrails import QuestionCheck, check_question
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)

Check = Callable[[str], QuestionCheck]


def _rate(hit: int, n: int) -> float | None:
    return round(hit / n, 3) if n else None


def score_rows(rows: Iterable[Mapping[str, Any]], check: Check = check_question) -> dict[str, Any]:
    """Hit rates per kind and group. For benign rows a hit is a false positive."""
    by_kind: dict[str, dict[str, Any]] = {}
    totals = {"injection": [0, 0], "pii": [0, 0], "benign": [0, 0]}
    missed, false_pos = [], []
    for row in rows:
        result = check(row["text"])
        group, kind = row["group"], row["kind"]
        if group == "injection":
            hit = result.blocked
        elif group == "pii":
            hit = f"pii:{kind}" in result.fired
        else:
            hit = bool(result.fired)
        if hit and group == "benign":
            false_pos.append(row["id"])
        elif not hit and group != "benign":
            missed.append(row["id"])
        k = by_kind.setdefault(kind, {"group": group, "n": 0, "hit": 0})
        k["n"], k["hit"] = k["n"] + 1, k["hit"] + int(hit)
        totals[group][0] += int(hit)
        totals[group][1] += 1
    for k in by_kind.values():
        k["rate"] = _rate(k["hit"], k["n"])
    return {
        "by_kind": by_kind,
        "injection_detection_rate": _rate(*totals["injection"]),
        "pii_detection_rate": _rate(*totals["pii"]),
        "benign_false_positive_rate": _rate(*totals["benign"]),
        "missed": missed,
        "false_positives": false_pos,
    }


def false_positives(texts: Mapping[str, str], check: Check = check_question) -> dict[str, Any]:
    """Clean texts that triggered any guard, by id, with what fired."""
    flagged = {key: fired for key, text in texts.items() if (fired := check(text).fired)}
    return {
        "n": len(texts),
        "flagged": len(flagged),
        "rate": _rate(len(flagged), len(texts)),
        "ids": flagged,
    }


def latency_profile(
    texts: list[str], repeats: int = 50, check: Check = check_question
) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        for text in texts:
            t0 = time.perf_counter()
            check(text)
            samples.append((time.perf_counter() - t0) * 1e6)
    return {
        "n": len(samples),
        "p50_us": round(_percentile(samples, 50), 1),
        "p95_us": round(_percentile(samples, 95), 1),
        "max_us": round(max(samples), 1),
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _corpus_texts(path: Path) -> dict[str, str]:
    from legalrag.ingest.validate import load_articles

    texts = {}
    for a in load_articles(path):
        for lang, text in (("ar", a.text_ar), ("en", a.text_en), ("note", a.note)):
            if text:
                texts[f"art{a.article_number}-{lang}"] = text
    return texts


def evaluate(attack_set: Path, golden: list[Path], corpus: Path | None) -> dict[str, Any]:
    rows = _jsonl(attack_set)
    items = [it for path in golden for it in _jsonl(path)]
    attacks = [
        {"id": it["id"], "group": "injection", "kind": "golden_injection", "text": it["question"]}
        for it in items
        if it["category"] == "injection"
    ]
    clean = {it["id"]: it["question"] for it in items if it["category"] != "injection"}
    summary: dict[str, Any] = {
        "attack_set": score_rows(rows + attacks),
        "clean_questions": false_positives(clean),
        "latency": latency_profile(list(clean.values()) + [r["text"] for r in rows]),
    }
    if corpus is not None and corpus.is_file():
        summary["corpus"] = false_positives(_corpus_texts(corpus))
    else:  # the corpus is DVC data, missing until `dvc pull`
        log.warning("corpus_skipped", path=str(corpus))
    return summary


def _metrics(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    a = summary["attack_set"]
    flat = {
        "injection_detection_rate": a["injection_detection_rate"],
        "pii_detection_rate": a["pii_detection_rate"],
        "benign_tricky_fpr": a["benign_false_positive_rate"],
        "clean_question_fpr": summary["clean_questions"]["rate"],
        "corpus_fpr": summary.get("corpus", {}).get("rate"),
        "latency_p95_us": summary["latency"]["p95_us"],
        # a hit on a benign kind is a false positive
        **{
            f"{'fp' if v['group'] == 'benign' else 'detect'}.{k}": v["rate"]
            for k, v in a["by_kind"].items()
        },
    }
    return {"guardrails": flat}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure the input guardrails")
    parser.add_argument("--attacks", default="data/golden/guardrails.jsonl")
    parser.add_argument(
        "--golden", nargs="+", default=["data/golden/golden_set.jsonl", "data/golden/heldout.jsonl"]
    )
    parser.add_argument("--corpus", default="data/processed/articles.json")
    parser.add_argument("--out", default="reports/eval/guardrails")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args(argv)
    configure_logging("INFO")
    summary = evaluate(Path(args.attacks), [Path(p) for p in args.golden], Path(args.corpus))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "summary.json"
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    log.info("guardrails_eval", **_metrics(summary)["guardrails"])
    if args.mlflow:
        from legalrag.eval.run import _md5
        from legalrag.eval.track import log_eval_run
        from legalrag.settings import get_settings

        log_eval_run(
            get_settings().mlflow_experiment,
            "guardrails",
            params={"guard": "regex", "attack_set": args.attacks},
            summary=_metrics(summary),
            tags={"eval_mode": "guardrails", "attack_set_md5": _md5(args.attacks)},
            artifacts=[path],
        )


if __name__ == "__main__":
    main()
