"""Decide whether a candidate run may replace production: no metric worse than production
minus a margin, in either language, and the same question set without errors.

    uv run python -m legalrag.eval.compare --candidate reports/eval/cand/metrics.json \
        --production reports/eval/prod/metrics.json --margin 0.02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GROUPS = ("all", "ar", "en")
METRICS = ("hit_at_1", "mrr")


def compare_runs(
    candidate: dict[str, Any], production: dict[str, Any], margin: float = 0.02
) -> dict[str, Any]:
    reasons = []
    for group in GROUPS:
        cand, prod = candidate[group], production[group]
        if cand["errors"]:
            reasons.append(f"{group}: {cand['errors']} questions failed")
        if cand["n"] != prod["n"]:
            reasons.append(f"{group}: {cand['n']} questions, production had {prod['n']}")
        for metric in METRICS:
            if cand[metric] < prod[metric] - margin:
                reasons.append(
                    f"{group} {metric} {cand[metric]:.3f} < production {prod[metric]:.3f} - {margin}"
                )
    summary = {g: {m: candidate[g][m] for m in METRICS} for g in GROUPS}
    return {"promote": not reasons, "reasons": reasons, "candidate": summary}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="compare a candidate eval run with production")
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--production", type=Path, required=True)
    p.add_argument("--margin", type=float, default=0.02)
    args = p.parse_args(argv)
    verdict = compare_runs(
        json.loads(args.candidate.read_text(encoding="utf-8")),
        json.loads(args.production.read_text(encoding="utf-8")),
        args.margin,
    )
    print(json.dumps(verdict))  # noqa: T201 - read by the Airflow branch task


if __name__ == "__main__":
    main()
