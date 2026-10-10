"""Time how long a label move takes to reach the API (lab drill, not a release tool).

    uv run python monitoring/prompt_label_drill.py --to 2 --back 1

Moves the ``production`` label to version ``--to``, polls GET /metadata once a second until the
API reports it, then moves it back to ``--back`` and times that too. It uses the raw label move
on purpose, to measure the mechanism; releases go through ``prompts promote`` / ``rollback``,
which check the quality gate. The original label is restored even if the drill fails halfway.
Writes reports/monitoring/prompt_label_drill.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from legalrag.observability.prompts import move_label

OUT = Path("reports/monitoring/prompt_label_drill.json")


def served(api: str) -> str:
    meta = httpx.get(f"{api}/metadata", timeout=10).raise_for_status().json()
    return f"{meta['prompt_version']} ({meta['prompt_source']})"


def step(client: Any, api: str, version: int, timeout_s: float = 180) -> dict[str, Any]:
    before = served(api)
    t0 = time.perf_counter()
    move_label(client, version=version, label="production")
    client.flush()
    while time.perf_counter() - t0 < timeout_s:
        now = served(api)
        if now != before:
            return {"to_version": version, "from": before, "to": now,
                    "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
        time.sleep(1)
    return {"to_version": version, "from": before, "to": None, "seconds": None}


def main() -> None:
    from langfuse import Langfuse

    from legalrag.settings import get_settings

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--api", default="http://127.0.0.1:8010")
    p.add_argument("--to", type=int, required=True, help="version to move production to")
    p.add_argument("--back", type=int, required=True, help="version production had before")
    args = p.parse_args()
    s = get_settings()
    client = Langfuse(public_key=s.langfuse_public_key.get_secret_value(),
                      secret_key=s.langfuse_secret_key.get_secret_value(), base_url=s.langfuse_host)  # fmt: skip
    results = []
    try:
        results.append(step(client, args.api, args.to))
        results.append(step(client, args.api, args.back))
    finally:  # never leave production on the drill's version
        move_label(client, version=args.back, label="production")
        client.flush()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {"run_at": datetime.now(UTC).isoformat(), "cache_ttl_s": 60, "steps": results}
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
