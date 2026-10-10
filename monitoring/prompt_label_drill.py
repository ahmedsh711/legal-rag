"""Promote a prompt by moving its label, then roll back, and time both from the API's side.

    uv run python monitoring/prompt_label_drill.py --promote 2 --rollback 1

Each step moves the ``production`` label in Langfuse, then polls GET /metadata once a second until
the API reports the new prompt version. The time measured is what a rollback really costs:
label move + the SDK prompt cache (60 s) + one request to notice. No build, no restart.
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

from legalrag.observability.prompts import move_label


def served(api: str) -> str:
    meta = httpx.get(f"{api}/metadata", timeout=10).json()
    return f"{meta['prompt_version']} ({meta['prompt_source']})"


def step(client, api: str, version: int, timeout_s: float = 180) -> dict:
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
    p.add_argument("--promote", type=int, required=True)
    p.add_argument("--rollback", type=int, required=True)
    args = p.parse_args()
    s = get_settings()
    client = Langfuse(public_key=s.langfuse_public_key.get_secret_value(),
                      secret_key=s.langfuse_secret_key.get_secret_value(), base_url=s.langfuse_host)  # fmt: skip
    results = [step(client, args.api, args.promote), step(client, args.api, args.rollback)]
    print(json.dumps(results, indent=2))  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
