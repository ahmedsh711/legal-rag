"""How much can we trust an LLM judge? Calibrate it against human labels, and probe its biases.

1. ``sheet``: write ``data/golden/to_label.csv`` with 20 answered questions (10 AR, 10 EN). A
   human marks each answer ``1`` (every legal claim is supported by the shown articles) or ``0``.
2. ``calibrate``: compare the human labels with
   - our judge (one JSON call: "is every legal claim supported?") -> agreement + Cohen's kappa,
   - RAGAS faithfulness >= 0.8 (same question, different judge prompt),
   - the same judge prompt run by the generator's model family -> self-preference bias,
   - our judge after appending a harmless sentence to each answer -> verbosity bias
     (a fair judge does not change its verdict because an answer got longer).

Kappa corrects agreement for chance: 0 = no better than chance, 1 = perfect. Position bias
(swapping A/B in pairwise judging) does not apply to a single-answer judge, so it is not probed.
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

LABEL = "supported (1/0)"
PADDING = {
    "en": " This is an important question that many people ask about Egyptian civil law.",
    "ar": " وهذا سؤال مهم يسأله كثير من الناس عن القانون المدني المصري.",
}
SYSTEM = """You check answers about the Egyptian Civil Code against the articles they were
based on. An answer is SUPPORTED only if every legal claim in it is stated in, or directly follows
from, the given articles. Ignore generic remarks that make no claim about the law. Citations like
[Art. N] are not claims. Reply with JSON only:
{"supported": true or false, "unsupported_claims": ["..."]}"""


class Judge:
    def __init__(self, client: Any, model: str):
        self.client, self.model = client, model

    async def supported(self, p: Prediction, pad: bool = False) -> int:
        answer = p.answer + (PADDING.get(p.lang, PADDING["en"]) if pad else "")
        articles = "\n\n".join(p.context_texts)
        user = f"<articles>\n{articles}\n</articles>\n\nQuestion: {p.question}\n\nAnswer: {answer}"
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=600,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return 1 if data.get("supported") is True else 0


def write_label_sheet(preds: Sequence[Prediction], path: str | Path, n: int = 20,
                      seed: int = 7) -> list[str]:  # fmt: skip
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
            w.writerow({"id": p.id, "lang": p.lang, "question": p.question,
                        "articles": "\n\n".join(p.context_texts), "answer": p.answer,
                        LABEL: "", "note": ""})  # fmt: skip
    return [p.id for p in chosen]


def read_labels(path: str | Path) -> dict[str, int]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return {r["id"]: int(r[LABEL]) for r in csv.DictReader(f) if r.get(LABEL, "").strip()}


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
    return dict(zip([p.id for p in preds], await asyncio.gather(
        *(judge.supported(p, pad=pad) for p in preds)), strict=True))  # fmt: skip


async def calibrate(preds: Sequence[Prediction], human: Mapping[str, int], judge: Judge,
                    same_family: Judge, ragas: Mapping[str, float]) -> dict[str, Any]:  # fmt: skip
    labelled = [p for p in preds if p.id in human]
    ours = await _verdicts(judge, labelled)
    padded = await _verdicts(judge, labelled, pad=True)
    family = await _verdicts(same_family, labelled)
    ragas_vote = {i: int(v >= 0.8) for i, v in ragas.items() if i in human}
    return {
        "n_labelled": len(labelled),
        "human_supported_rate": sum(human[p.id] for p in labelled) / max(len(labelled), 1),
        "judge": {"model": judge.model, "agreement": agreement(human, ours),
                  "kappa": cohen_kappa(human, ours)},
        "ragas_faithfulness_ge_0_8": {"agreement": agreement(human, ragas_vote),
                                      "kappa": cohen_kappa(human, ragas_vote)},
        "self_preference_probe": {"model": same_family.model,
                                  "agreement": agreement(human, family),
                                  "kappa": cohen_kappa(human, family),
                                  "supported_rate": sum(family.values()) / max(len(family), 1),
                                  "judge_supported_rate": sum(ours.values()) / max(len(ours), 1)},
        "verbosity_probe": {"verdict_flips": sum(ours[i] != padded[i] for i in ours),
                            "of": len(ours)},
    }  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LLM-judge label sheet and calibration")
    p.add_argument("command", choices=["sheet", "calibrate"])
    p.add_argument("--predictions", default="reports/eval/baseline/predictions.jsonl")
    p.add_argument("--labels", default="data/golden/to_label.csv")
    p.add_argument("--ragas", default="reports/eval/baseline/ragas.jsonl")
    p.add_argument("--out", default="reports/eval/judge_calibration.json")
    args = p.parse_args(argv)

    from legalrag.eval.run import read_predictions

    preds = read_predictions(args.predictions)
    if args.command == "sheet":
        write_label_sheet(preds, args.labels)
        return
    from openai import AsyncOpenAI

    from legalrag.settings import get_settings

    s = get_settings()
    judge_client = AsyncOpenAI(base_url=s.judge_base_url, api_key=s.judge_api_key, timeout=120,
                               max_retries=4)  # fmt: skip
    gen_client = AsyncOpenAI(base_url=s.active_llm_base_url, api_key=s.active_llm_api_key,
                             timeout=120, max_retries=4)  # fmt: skip
    rows = [json.loads(x) for x in Path(args.ragas).read_text(encoding="utf-8").splitlines() if x]
    ragas = {r["id"]: r["faithfulness"] for r in rows if r.get("faithfulness") is not None}
    report = asyncio.run(calibrate(
        preds, read_labels(args.labels), Judge(judge_client, s.judge_model),
        Judge(gen_client, s.active_llm_model), ragas))  # fmt: skip
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
