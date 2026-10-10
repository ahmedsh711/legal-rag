"""Online evaluation: judge a sample of real answers every night, not only the golden set.

    uv run --group eval python -m legalrag.monitoring.online_eval --hours 24 --rate 0.05

The golden set says how good the system was on questions we wrote. Real traffic drifts away from
them, so each night this job takes a random 5 % of yesterday's generations from Langfuse (the
prompt with its articles, and the raw model output), has the RAGAS judge score their
faithfulness, and writes the result two places:
- on each trace in Langfuse, as a ``faithfulness`` score (find the bad answers by clicking);
- the mean as ``rag_eval_faithfulness`` in the node-exporter textfile, where Prometheus picks it
  up and the ``FaithfulnessLow`` alert (below 0.80) watches it.
Refusals are skipped (there is no claim to check). The raw model output is scored, even when the
citation check later replaced it: the question is "does the model stay grounded?".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from legalrag.generation import REFUSAL_AR, REFUSAL_EN, detect_language
from legalrag.logging_conf import configure_logging, get_logger
from legalrag.monitoring.files import write_atomic

log = get_logger(__name__)

_ARTICLES = re.compile(r"<articles>\n(.*)\n</articles>", re.DOTALL)
_QUESTION = re.compile(r"\nQuestion: (.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Judged:
    trace_id: str
    observation_id: str
    question: str
    contexts: list[str]
    answer: str


def parse_generation(gen: Mapping[str, Any]) -> Judged | None:
    """Question, articles and answer back out of a traced generation (None if not judgeable)."""
    answer = (gen.get("output") or "").strip()
    if not answer or answer in (REFUSAL_EN, REFUSAL_AR):
        return None
    try:
        messages = json.loads(gen["input"]) if isinstance(gen["input"], str) else gen["input"]
        user = next(m["content"] for m in messages if m["role"] == "user")
    except (TypeError, ValueError, KeyError, StopIteration):
        return None
    articles, question = _ARTICLES.search(user), _QUESTION.search(user)
    if not (articles and question):
        return None
    contexts = re.split(r"\n\n(?=\[Art\. )", articles.group(1))
    return Judged(gen["traceId"], gen["id"], question.group(1).strip(), contexts, answer)


def sample(generations: Sequence[Mapping[str, Any]], rate: float, seed: int) -> list[Any]:
    k = round(len(generations) * rate)
    return (
        random.Random(seed).sample(list(generations), k)
        if k < len(generations)
        else list(generations)
    )


async def run_online_eval(generations: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any],
                          client: Any, rate: float = 0.05, seed: int = 0,
                          rpm: float | None = None) -> tuple[dict[str, Any], list[str]]:  # fmt: skip
    from legalrag.eval.metrics import Prediction
    from legalrag.eval.ragas_run import score_predictions

    # parse first, then sample: the rate is a share of answers that can be judged (review)
    judgeable = [j for g in generations if (j := parse_generation(g))]
    picked = sample(judgeable, rate, seed)
    preds = [Prediction(id=j.observation_id, lang=detect_language(j.question),
                        category="in_scope", question=j.question,
                        gold_articles=[], answer=j.answer, refused=False, cited=[],
                        context_articles=[], context_texts=j.contexts) for j in picked]  # fmt: skip
    rows = await score_predictions(preds, {"faithfulness": metrics["faithfulness"]}, 2, rpm)
    by_id = {j.observation_id: j for j in picked}
    values = []
    for row in rows:
        if (value := row.get("faithfulness")) is None:
            continue
        j = by_id[row["id"]]
        client.create_score(name="faithfulness", value=value, trace_id=j.trace_id,
                            observation_id=j.observation_id, data_type="NUMERIC",
                            comment="online sample, RAGAS")  # fmt: skip
        values.append(value)
    client.flush()
    mean = round(sum(values) / len(values), 3) if values else None
    unscored = len(picked) - len(values)  # the judge failed: reported, not silently dropped
    summary = {"sampled": len(picked), "scored": len(values), "unscored": unscored,
               "faithfulness": mean}  # fmt: skip
    lines = ["# HELP rag_eval_samples Answers judged in the last online evaluation",
             "# TYPE rag_eval_samples gauge", f"rag_eval_samples {len(values)}",
             "# HELP rag_eval_unscored Sampled answers the judge could not score",
             "# TYPE rag_eval_unscored gauge", f"rag_eval_unscored {unscored}"]  # fmt: skip
    if mean is not None:
        lines += ["# HELP rag_eval_faithfulness Mean RAGAS faithfulness of sampled real answers",
                  "# TYPE rag_eval_faithfulness gauge", f"rag_eval_faithfulness {mean}"]  # fmt: skip
    return summary, lines


def fetch_generations(base_url: str, auth: tuple[str, str], since: datetime,
                      limit: int = 5000) -> list[dict[str, Any]]:  # fmt: skip
    """Generations since ``since`` from the Langfuse v2 observations API (cursor pages)."""
    import httpx

    out: list[dict[str, Any]] = []
    params: dict[str, Any] = {"type": "GENERATION", "fields": "core,io", "limit": 100,
                              "fromStartTime": since.isoformat()}  # fmt: skip
    with httpx.Client(base_url=base_url, auth=auth, timeout=30) as http:
        while len(out) < limit:
            page = http.get("/api/public/v2/observations", params=params).raise_for_status().json()
            out += page.get("data", [])
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor or not page.get("data"):
                break
            params["cursor"] = cursor
    return out[:limit]


def main(argv: list[str] | None = None) -> None:
    from langfuse import Langfuse

    from legalrag.eval.ragas_run import make_metrics
    from legalrag.settings import get_settings

    p = argparse.ArgumentParser(description="nightly online evaluation of sampled answers")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--rate", type=float, default=0.05)
    p.add_argument("--textfile", default="monitoring/textfile/online_eval.prom")
    args = p.parse_args(argv)
    configure_logging("INFO")
    s = get_settings()
    auth = (s.langfuse_public_key.get_secret_value(), s.langfuse_secret_key.get_secret_value())
    gens = fetch_generations(s.langfuse_host, auth, datetime.now(UTC) - timedelta(hours=args.hours))
    metrics = make_metrics(s.judge_base_url, s.judge_api_key, s.judge_model, s.judge_embedding_model,
                           names=["faithfulness"], reasoning_effort=s.judge_reasoning_effort)  # fmt: skip
    client = Langfuse(public_key=auth[0], secret_key=auth[1], base_url=s.langfuse_host)
    seed = int(datetime.now(UTC).strftime("%Y%m%d"))  # a new sample every day, reproducible
    summary, lines = asyncio.run(run_online_eval(gens, metrics, client, args.rate, seed,
                                                 rpm=s.judge_rpm))  # fmt: skip
    write_atomic(Path(args.textfile), "\n".join(lines) + "\n")
    log.info("online_eval_done", generations=len(gens), **summary)


if __name__ == "__main__":
    main()
