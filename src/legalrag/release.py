"""Canary rollout behind nginx: set the canary's share, roll back, or watch and roll back.

    uv run python -m legalrag.release set 5          # 5 % of requests to the canary
    uv run python -m legalrag.release rollback       # everything back to stable, timed
    uv run python -m legalrag.release watch          # roll back when the canary misbehaves

The share lives in serving/nginx/release/split.conf (an nginx split_clients block); changing
it is a config reload, so in-flight requests finish and nothing restarts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)

SPLIT = Path("serving/nginx/release/split.conf")
NGINX = "legal-rag-nginx-1"
FRONT = "http://127.0.0.1:8088"
PROMETHEUS = "http://127.0.0.1:9090"
WINDOW = "2m"


@dataclass(frozen=True)
class CanaryHealth:
    requests: float
    error_ratio: float
    p95_seconds: float | None


@dataclass(frozen=True)
class Limits:
    min_requests: int = 10
    max_error_ratio: float = 0.05
    max_p95_seconds: float = 5.0  # the /ask SLO


def render_split(percent: int) -> str:
    if not 0 <= percent <= 100:
        raise ValueError(f"canary share must be 0-100, got {percent}")
    if percent == 0:
        body = "    * stable;"
    elif percent == 100:
        body = "    * canary;"
    else:
        body = f"    {percent}% canary;\n    * stable;"
    return 'split_clients "${request_id}" $release {\n' + body + "\n}\n"


def should_roll_back(health: CanaryHealth, limits: Limits) -> list[str]:
    if health.requests < limits.min_requests:
        return []
    reasons = []
    if health.error_ratio > limits.max_error_ratio:
        reasons.append(f"5xx ratio {health.error_ratio:.3f} > {limits.max_error_ratio}")
    if health.p95_seconds is not None and health.p95_seconds > limits.max_p95_seconds:
        reasons.append(f"p95 {health.p95_seconds:.2f} s > {limits.max_p95_seconds} s")
    return reasons


def set_share(percent: int, split: Path = SPLIT, container: str = NGINX) -> None:
    split.parent.mkdir(parents=True, exist_ok=True)
    tmp = split.with_suffix(".new")
    tmp.write_text(render_split(percent), encoding="utf-8", newline="\n")
    tmp.replace(split)
    subprocess.run(["docker", "exec", container, "nginx", "-t", "-q"], check=True)
    subprocess.run(["docker", "exec", container, "nginx", "-s", "reload"], check=True)
    log.info("canary_share_set", percent=percent)


def rollback(front: str = FRONT, streak: int = 100, timeout_s: float = 60) -> float:
    """Send everything to stable; seconds until `streak` responses in a row came from stable."""
    t0 = time.perf_counter()
    set_share(0)
    in_a_row = 0
    with httpx.Client(timeout=5) as client:
        while in_a_row < streak:
            if time.perf_counter() - t0 > timeout_s:
                raise RuntimeError(
                    f"no {streak} stable responses within {timeout_s} s; check nginx"
                )
            try:
                release = client.get(f"{front}/live").headers.get("x-release")
            except httpx.TransportError:  # nginx closes idle keep-alive connections on reload
                release = None
            in_a_row = in_a_row + 1 if release == "stable" else 0
    seconds = time.perf_counter() - t0
    log.info("rolled_back", seconds=round(seconds, 2), verified_requests=streak)
    return seconds


def _scalar(client: httpx.Client, prometheus: str, query: str) -> float | None:
    result = client.get(f"{prometheus}/api/v1/query", params={"query": query}).json()["data"]
    values = [float(r["value"][1]) for r in result["result"]]
    return values[0] if values and values[0] == values[0] else None  # NaN -> None


def canary_health(client: httpx.Client, prometheus: str = PROMETHEUS) -> CanaryHealth:
    sel = 'job="api-canary",endpoint="ask"'
    total = f"sum(increase(rag_requests_total{{{sel}}}[{WINDOW}]))"
    errors = f'sum(increase(rag_requests_total{{{sel},status=~"5.."}}[{WINDOW}]))'
    p95 = (
        f"histogram_quantile(0.95, sum by (le) "
        f"(rate(rag_request_seconds_bucket{{{sel}}}[{WINDOW}])))"
    )
    requests = _scalar(client, prometheus, total) or 0.0
    failed = _scalar(client, prometheus, errors) or 0.0
    return CanaryHealth(
        requests=requests,
        error_ratio=failed / requests if requests else 0.0,
        p95_seconds=_scalar(client, prometheus, p95),
    )


def watch(limits: Limits, interval_s: float = 15, max_checks: int | None = None) -> bool:
    """Check the canary every `interval_s`; roll back and return True on the first breach."""
    checks = 0
    with httpx.Client(timeout=10) as client:
        while max_checks is None or checks < max_checks:
            try:
                health = canary_health(client)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                log.warning("canary_check_failed", error=type(exc).__name__)
                time.sleep(interval_s)
                continue
            reasons = should_roll_back(health, limits)
            log.info("canary_check", **health.__dict__, reasons=reasons)
            if reasons:
                result = {"rolled_back": True, "reasons": reasons, "seconds": round(rollback(), 2)}
                print(json.dumps(result))  # noqa: T201
                return True
            checks += 1
            time.sleep(interval_s)
    return False


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="canary rollout behind nginx")
    sub = p.add_subparsers(dest="command", required=True)
    share = sub.add_parser("set", help="share of requests for the canary, 0-100")
    share.add_argument("percent", type=int)
    sub.add_parser("rollback")
    w = sub.add_parser("watch")
    w.add_argument("--interval", type=float, default=15)
    w.add_argument("--max-error-ratio", type=float, default=Limits.max_error_ratio)
    w.add_argument("--max-p95", type=float, default=Limits.max_p95_seconds)
    w.add_argument("--min-requests", type=int, default=Limits.min_requests)
    args = p.parse_args(argv)
    configure_logging("INFO")
    if args.command == "set":
        set_share(args.percent)
    elif args.command == "rollback":
        print(json.dumps({"rolled_back": True, "seconds": round(rollback(), 2)}))  # noqa: T201
    else:
        watch(Limits(args.min_requests, args.max_error_ratio, args.max_p95), args.interval)


if __name__ == "__main__":
    main()
