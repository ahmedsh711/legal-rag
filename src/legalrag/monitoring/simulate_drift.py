"""Make drift happen on purpose, so we can check the drift job sees it (and stays quiet otherwise).

    uv run python -m legalrag.monitoring.simulate_drift control --n 200   # golden-like traffic
    uv run python -m legalrag.monitoring.simulate_drift intent  --n 200   # topic shift

- ``control``: questions drawn from the golden set (the reference traffic). The drift job must
  NOT fire on it: that is its false-alarm check.
- ``intent``: a query-intent shift. People start asking about family law, crime, tax, labour and
  traffic (outside the Civil Code), mostly in Arabic. Retrieval lands in other books, the gate
  and the citation check refuse more, the language mix moves. The drift job must fire.
Both send real requests to the API (one issued key per worker), so the events, metrics and
alerts are the ones production would produce.
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
    "ما هي إجراءات الطلاق للضرر في محكمة الأسرة؟", "ما هي مدة حضانة الأم للأطفال بعد الطلاق؟",
    "كيف تحسب النفقة الزوجية بعد الخلع؟", "ما عقوبة السرقة بالإكراه في قانون العقوبات؟",
    "كم سنة سجن عقوبة خيانة الأمانة؟", "ما هي نسبة ضريبة القيمة المضافة على الخدمات؟",
    "كيف أقدم إقرار ضريبة الدخل إلكترونيا؟", "كم ساعة عمل إضافية يسمح بها قانون العمل؟",
    "ما هي غرامة تجاوز السرعة على الطريق الدائري؟", "كيف أجدد رخصة القيادة المنتهية؟",
    "ما شروط عقد الزواج العرفي؟", "هل يحق للزوجة طلب الخلع دون موافقة الزوج؟",
    "ما عقوبة تعاطي المخدرات لأول مرة؟", "كم مدة الإجازة السنوية للموظف في القطاع الخاص؟",
    "ما هي رسوم استخراج جواز السفر المستعجل؟", "كيف أسجل شركة ذات مسؤولية محدودة؟",
    "What is the penalty for drunk driving in Egypt?", "How is child custody decided after divorce?",
    "What is the income tax rate for freelancers?", "How many days of sick leave does labour law give?",
]  # fmt: skip


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
                r = await client.post(f"{url}/ask", json={"question": picks[i]},
                                      headers={"X-API-Key": keys[i % len(keys)]})  # fmt: skip
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
