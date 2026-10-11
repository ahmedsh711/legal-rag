"""Calibrate the LLM judge against human labels and probe its biases.

``sheet`` writes ``data/golden/to_label.csv`` (10 AR + 10 EN answers) for a human to mark 1/0.
``calibrate`` reports agreement and Cohen's kappa for the judge, for RAGAS faithfulness >= 0.8
and for the generator's model family (self-preference), plus verdict flips after padding each
answer with a neutral sentence (verbosity bias).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from legalrag.eval.metrics import Prediction
from legalrag.eval.pacing import Pacer

LABEL = "supported (1/0)"
PADDING = {
    "en": " This is an important question that many people ask about Egyptian civil law.",
    "ar": " وهذا سؤال مهم يسأله كثير من الناس عن القانون المدني المصري.",
}
# An unsupported (and false) legal claim; appended to a real answer it makes a known negative.
FABRICATED = {
    "en": " In addition, the parties may agree in writing to extend this period indefinitely.",
    "ar": " كما يجوز للطرفين الاتفاق كتابة على مد هذه المدة إلى أجل غير محدود.",
}
SYSTEM = """You check answers about the Egyptian Civil Code against the articles they were
based on. An answer is SUPPORTED only if every legal claim in it is stated in, or directly follows
from, the given articles. Ignore generic remarks that make no claim about the law. Citations like
[Art. N] are not claims. Reply with JSON only:
{"supported": true or false, "unsupported_claims": ["..."]}"""


class Judge:
    def __init__(self, client: Any, model: str, pacer: Pacer | None = None, **extra: Any):
        self.client, self.model = client, model
        self.pacer = pacer or Pacer(None)
        self.extra = extra  # e.g. reasoning_effort="none" for thinking models

    async def supported(self, p: Prediction, pad: bool = False) -> int | None:
        """1 supported, 0 not, None when the reply has no usable verdict."""
        await self.pacer.wait()
        answer = p.answer + (PADDING.get(p.lang, PADDING["en"]) if pad else "")
        articles = "\n\n".join(p.context_texts)
        user = f"<articles>\n{articles}\n</articles>\n\nQuestion: {p.question}\n\nAnswer: {answer}"
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=600,
            **self.extra,
        )
        return _verdict(resp.choices[0].message.content or "")


def _verdict(text: str) -> int | None:
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(body).get("supported")
    except (json.JSONDecodeError, AttributeError):  # prose, truncated JSON, a list...
        return None
    return int(value) if isinstance(value, bool) else None


def write_label_sheet(
    preds: Sequence[Prediction], path: str | Path, n: int = 20, seed: int = 7
) -> list[str]:
    """``n`` answered items, half Arabic half English, as a CSV that opens cleanly in Excel."""
    rng = random.Random(seed)
    chosen: list[Prediction] = []
    for lang in ("ar", "en"):
        pool = [p for p in preds if p.lang == lang and p.answerable and not p.refused]
        chosen += rng.sample(pool, min(n // 2, len(pool)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:  # BOM for Excel
        w = csv.DictWriter(f, ["id", "lang", "question", "articles", "answer", LABEL, "note"])
        w.writeheader()
        for p in chosen:
            w.writerow(
                {
                    "id": p.id,
                    "lang": p.lang,
                    "question": p.question,
                    "articles": "\n\n".join(p.context_texts),
                    "answer": p.answer,
                    LABEL: "",
                    "note": "",
                }
            )
    return [p.id for p in chosen]


YES, NO = {"1", "1.0", "true", "yes", "y"}, {"0", "0.0", "false", "no", "n"}


def read_labels(path: str | Path) -> dict[str, int]:
    """Human labels; blank rows are skipped, Excel spellings (1.0, TRUE, yes) are accepted."""
    labels = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get(LABEL, "").strip().lower()
            if not raw:
                continue
            if raw not in YES | NO:
                raise ValueError(f"row {row['id']}: label {raw!r} is not 1/0")
            labels[row["id"]] = int(raw in YES)
    return labels


def agreement(human: Mapping[str, int], other: Mapping[str, int]) -> float:
    ids = human.keys() & other.keys()
    return sum(human[i] == other[i] for i in ids) / len(ids) if ids else float("nan")


def cohen_kappa(human: Mapping[str, int], other: Mapping[str, int]) -> float:
    ids = sorted(human.keys() & other.keys())
    if not ids:
        return float("nan")
    po = agreement(human, other)
    p_h = sum(human[i] for i in ids) / len(ids)
    p_o = sum(other[i] for i in ids) / len(ids)
    pe = p_h * p_o + (1 - p_h) * (1 - p_o)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


async def _verdicts(judge: Judge, preds: Sequence[Prediction], pad: bool = False) -> dict:
    """Unparseable verdicts are dropped, not counted as 0."""
    votes = await asyncio.gather(*(judge.supported(p, pad=pad) for p in preds))
    return {p.id: v for p, v in zip(preds, votes, strict=True) if v is not None}


def with_fabricated_claim(p: Prediction) -> Prediction:
    """Same question and articles, plus one unsupported legal claim: a known negative."""
    return p.model_copy(
        update={"id": f"{p.id}-neg", "answer": p.answer + FABRICATED.get(p.lang, FABRICATED["en"])}
    )


def _scores(truth: Mapping[str, int], votes: Mapping[str, int]) -> dict[str, float]:
    return {
        "agreement": agreement(truth, votes),
        "kappa": cohen_kappa(truth, votes),
        "n": len(truth.keys() & votes.keys()),
    }


async def calibrate(
    preds: Sequence[Prediction],
    human: Mapping[str, int],
    judge: Judge,
    same_family: Judge | None,
    ragas: Mapping[str, float],
) -> dict[str, Any]:
    """Score against human labels plus one synthetic negative per labelled answer.

    The negatives keep kappa meaningful when humans mark every real answer as supported."""
    labelled = [p for p in preds if p.id in human]
    negatives = [with_fabricated_claim(p) for p in labelled]
    truth = {**human, **{n.id: 0 for n in negatives}}
    ours = await _verdicts(judge, labelled + negatives)
    padded = await _verdicts(judge, labelled, pad=True)
    both = [p.id for p in labelled if p.id in ours and p.id in padded]
    ragas_vote = {i: int(v >= 0.8) for i, v in ragas.items() if i in truth}
    report: dict[str, Any] = {
        "n_labelled": len(labelled),
        "n_synthetic_negatives": len(negatives),
        "human_supported_rate": sum(human[p.id] for p in labelled) / max(len(labelled), 1),
        "judge": {"model": judge.model, **_scores(truth, ours)},
        "judge_on_real_answers_only": _scores(human, ours),
        "ragas_faithfulness_ge_0_8": _scores(truth, ragas_vote),
        "verbosity_probe": {
            "verdict_flips": sum(ours[i] != padded[i] for i in both),
            "of": len(both),
        },
        "unparseable_verdicts": len(labelled) + len(negatives) - len(ours),
    }
    if same_family is None or same_family.model == judge.model:
        report["self_preference_probe"] = "not measured: judge and generator share a model"
    else:
        family = await _verdicts(same_family, labelled + negatives)
        report["self_preference_probe"] = {"model": same_family.model, **_scores(truth, family)}
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LLM-judge label sheet and calibration")
    p.add_argument("command", choices=["sheet", "calibrate"])
    p.add_argument("--predictions", default="reports/eval/baseline/predictions.jsonl")
    p.add_argument("--labels", default="data/golden/to_label.csv")
    p.add_argument("--out", default="reports/eval/judge_calibration.json")
    p.add_argument("--mlflow", action="store_true", help="log the calibration as an MLflow run")
    args = p.parse_args(argv)

    from legalrag.eval.run import read_predictions

    preds = read_predictions(args.predictions)
    if args.command == "sheet":
        write_label_sheet(preds, args.labels)
        return
    report = asyncio.run(_calibrate_cli(preds, read_labels(args.labels)))
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.mlflow:
        _log_calibration(report, args.out)


async def _calibrate_cli(preds: Sequence[Prediction], human: Mapping[str, int]) -> dict:
    from openai import AsyncOpenAI

    from legalrag.eval.ragas_run import make_metrics, score_predictions
    from legalrag.logging_conf import configure_logging
    from legalrag.settings import get_settings

    s = get_settings()
    configure_logging(s.log_level)
    pacer = Pacer(s.judge_rpm)  # one per-minute budget for both judges
    effort = {"reasoning_effort": s.judge_reasoning_effort} if s.judge_reasoning_effort else {}
    client = AsyncOpenAI(
        base_url=s.judge_base_url, api_key=s.judge_api_key, timeout=120, max_retries=4
    )
    gen = AsyncOpenAI(
        base_url=s.active_llm_base_url, api_key=s.active_llm_api_key, timeout=120, max_retries=4
    )
    labelled = [p for p in preds if p.id in human]
    faith = make_metrics(
        s.judge_base_url,
        s.judge_api_key,
        s.judge_model,
        s.judge_embedding_model,
        names=("faithfulness",),
        reasoning_effort=s.judge_reasoning_effort,
    )
    rows = await score_predictions(
        labelled + [with_fabricated_claim(p) for p in labelled],
        faith,
        concurrency=2,
        requests_per_minute=s.judge_rpm,
    )
    ragas = {r["id"]: r["faithfulness"] for r in rows if r.get("faithfulness") is not None}
    return await calibrate(
        preds,
        human,
        Judge(client, s.judge_model, pacer, **effort),
        Judge(gen, s.active_llm_model, pacer),
        ragas,
    )


def _log_calibration(report: Mapping[str, Any], path: str) -> None:
    import mlflow

    from legalrag.settings import get_settings

    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment)
    with mlflow.start_run(run_name="judge-calibration", tags={"eval_mode": "judge_calibration"}):
        mlflow.log_params(
            {
                "judge_model": report["judge"]["model"],
                "n_labelled": report["n_labelled"],
                "n_synthetic_negatives": report["n_synthetic_negatives"],
            }
        )
        for group in ("judge", "judge_on_real_answers_only", "ragas_faithfulness_ge_0_8"):
            for k in ("agreement", "kappa"):
                value = report[group][k]
                if value == value:  # skip NaN
                    mlflow.log_metric(f"{group}.{k}", value)
        mlflow.log_metric("verbosity_flips", report["verbosity_probe"]["verdict_flips"])
        mlflow.log_artifact(path)


if __name__ == "__main__":
    main()
