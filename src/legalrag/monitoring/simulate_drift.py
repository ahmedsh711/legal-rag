"""Send control or drifted traffic to the API to check the drift job end to end.

    uv run python -m legalrag.monitoring.simulate_drift control --n 200   # golden-like traffic
    uv run python -m legalrag.monitoring.simulate_drift intent  --n 200   # topic shift

``control`` replays golden questions and must not trigger drift. ``intent`` asks about family,
criminal, tax, labour and traffic law, mostly in Arabic, and must trigger it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import httpx

from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)

OFF_TOPIC = [
    "ما هي إجراءات الطلاق للضرر في محكمة الأسرة؟",
    "ما هي مدة حضانة الأم للأطفال بعد الطلاق؟",
    "كيف تحسب النفقة الزوجية بعد الخلع؟",
    "ما عقوبة السرقة بالإكراه في قانون العقوبات؟",
    "كم سنة سجن عقوبة خيانة الأمانة؟",
    "ما هي نسبة ضريبة القيمة المضافة على الخدمات؟",
    "كيف أقدم إقرار ضريبة الدخل إلكترونيا؟",
    "كم ساعة عمل إضافية يسمح بها قانون العمل؟",
    "ما هي غرامة تجاوز السرعة على الطريق الدائري؟",
    "كيف أجدد رخصة القيادة المنتهية؟",
    "ما شروط عقد الزواج العرفي؟",
    "هل يحق للزوجة طلب الخلع دون موافقة الزوج؟",
    "ما عقوبة تعاطي المخدرات لأول مرة؟",
    "كم مدة الإجازة السنوية للموظف في القطاع الخاص؟",
    "ما هي رسوم استخراج جواز السفر المستعجل؟",
    "كيف أسجل شركة ذات مسؤولية محدودة؟",
    "What is the penalty for drunk driving in Egypt?",
    "How is child custody decided after divorce?",
    "What is the income tax rate for freelancers?",
    "How many days of sick leave does labour law give?",
]


def golden_questions(path: Path) -> list[str]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r["question"] for r in rows if r["category"] != "injection"]


async def send(questions: list[str], n: int, url: str, keys: list[str], workers: int) -> dict:
    rng = random.Random(0)
    picks = [rng.choice(questions) for _ in range(n)]
    counts = {"ok": 0, "refused": 0, "error": 0}
    gate = asyncio.Semaphore(workers)

    async def one(i: int, client: httpx.AsyncClient) -> None:
        async with gate:
            try:
                r = await client.post(
                    f"{url}/ask",
                    json={"question": picks[i]},
                    headers={"X-API-Key": keys[i % len(keys)]},
                )
                r.raise_for_status()
                counts["refused" if r.json()["refused"] else "ok"] += 1
            except httpx.HTTPError as exc:
                counts["error"] += 1
                log.warning("simulate_request_failed", error=type(exc).__name__)

    async with httpx.AsyncClient(timeout=120) as client:
        await asyncio.gather(*(one(i, client) for i in range(n)))
    return counts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="send control or drifted traffic to the API")
    p.add_argument("kind", choices=["control", "intent"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--url", default="http://127.0.0.1:8010")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--golden", default="data/golden/golden_set.jsonl")
    args = p.parse_args(argv)
    configure_logging("INFO")
    keys = [k for k in os.environ.get("LOADTEST_API_KEYS", "").split(",") if k] or ["anonymous"]
    questions = golden_questions(Path(args.golden)) if args.kind == "control" else OFF_TOPIC
    counts = asyncio.run(send(questions, args.n, args.url, keys, args.workers))
    log.info("simulate_done", kind=args.kind, **counts)


if __name__ == "__main__":
    main()
