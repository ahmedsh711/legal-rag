"""Dashboard as code: writes monitoring/grafana/dashboards/legal-rag.json.

    uv run python monitoring/grafana/build_dashboard.py

Grafana loads that file at start-up (provisioning), so the dashboard lives in git, is reviewed
like code and survives `docker compose down -v`. Four rows, top to bottom in the order you
debug: is it serving (Service) -> is a box full (Resources) -> which stage or guard changed
(RAG behaviour) -> are the questions or the quality drifting (Drift & quality).
"""

from __future__ import annotations

import json
from pathlib import Path

PROM = {"type": "prometheus", "uid": "prometheus"}
OUT = Path(__file__).parent / "dashboards" / "legal-rag.json"


class Layout:
    """Places panels left to right, 24 columns wide, starting a new line when full."""

    def __init__(self) -> None:
        self.panels: list[dict] = []
        self.x = self.y = 0
        self.next_id = 1

    def _add(self, panel: dict, w: int, h: int) -> None:
        if self.x + w > 24:
            self.x, self.y = 0, self.y + h
        panel.update(id=self.next_id, gridPos={"x": self.x, "y": self.y, "w": w, "h": h})
        self.next_id += 1
        self.panels.append(panel)
        self.x += w

    def row(self, title: str) -> None:
        if self.x:
            self.x, self.y = 0, self.y + 8
        self._add({"type": "row", "title": title, "collapsed": False, "panels": []}, 24, 1)
        self.x, self.y = 0, self.y + 1

    def chart(self, title: str, targets: list[tuple[str, str]], unit: str = "short",
              w: int = 8, kind: str = "timeseries", thresholds: list | None = None) -> None:  # fmt: skip
        defaults: dict = {"unit": unit}
        if thresholds:
            defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
        panel = {
            "type": kind, "title": title, "datasource": PROM,
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "targets": [{"refId": chr(65 + i), "expr": expr, "legendFormat": legend,
                         "datasource": PROM} for i, (expr, legend) in enumerate(targets)],
        }  # fmt: skip
        self._add(panel, w, 8)


def build() -> dict:
    g = Layout()
    green_red = [{"color": "green", "value": None}]
    g.row("Service: is it answering, how fast, how often wrong?")
    g.chart(
        "/ask requests per second by status",
        [('sum by (status) (rag:request_rate:5m{endpoint="ask"})', "{{status}}")],
        "reqps",
    )
    g.chart("/ask latency p50 / p95 (SLO p95 < 5 s)",
            [('histogram_quantile(0.5, sum by (le) (rate(rag_request_seconds_bucket{endpoint="ask"}[5m])))', "p50"),
             ("rag:ask_latency_p95:5m", "p95")], "s")  # fmt: skip
    g.chart("5xx ratio (5 min)", [("rag:ask_error_ratio:5m", "5xx")], "percentunit", 4, "stat",
            green_red + [{"color": "red", "value": 0.01}])  # fmt: skip
    g.chart("Time to first token p95", [("rag:ttft_p95:5m", "ttft")], "s", 4, "stat",
            green_red + [{"color": "red", "value": 2}])  # fmt: skip
    g.chart("Requests in flight", [("sum(rag_inflight_requests)", "in flight")], "short", 6)
    g.chart("Health probe (blackbox)", [('probe_success{job="blackbox-health"}', "up")], "short",
            6, "stat", [{"color": "red", "value": None}, {"color": "green", "value": 1}])  # fmt: skip
    g.chart("What is serving", [("rag_info", "{{llm_model}} · prompt {{prompt_version}} · "
                                 "{{decider}} · {{index_collection}}")], "short", 12, "stat")  # fmt: skip

    g.row("Resources: which box is full?")
    g.chart(
        "Container CPU (cores)",
        [
            (
                'sum by (name) (rate(container_cpu_usage_seconds_total{name=~"legal-rag-.+"}[1m]))',
                "{{name}}",
            )
        ],
    )
    g.chart(
        "Container memory",
        [('sum by (name) (container_memory_working_set_bytes{name=~"legal-rag-.+"})', "{{name}}")],
        "bytes",
    )
    g.chart(
        "vLLM requests running / waiting",
        [
            ("sum(vllm:num_requests_running)", "running"),
            ("sum(vllm:num_requests_waiting)", "waiting"),
        ],
    )
    g.chart("vLLM KV cache used", [("max(vllm:kv_cache_usage_perc)", "kv cache")], "percentunit", 6)
    g.chart("Rate limiter", [("sum by (outcome) (rate(rag_ratelimit_total[5m]))", "{{outcome}}")],
            "reqps", 6)  # fmt: skip

    g.row("RAG behaviour: which stage, which guard?")
    g.chart("Stage p95", [("rag:stage_latency_p95:5m", "{{stage}}")], "s")
    g.chart(
        "Guards firing per minute",
        [("sum by (guard) (rate(rag_guardrail_total[5m])) * 60", "{{guard}}")],
    )
    g.chart("Refusal ratio (15 min)", [("rag:refusal_ratio:15m", "refused")], "percentunit")
    g.chart(
        "LLM tokens per hour ($backend)",
        [('sum by (backend, type) (rag:tokens:1h{backend=~"$backend"})', "{{backend}} {{type}}")],
    )
    g.chart(
        "Decider verdicts per minute",
        [
            (
                "sum by (decider, outcome) (rate(rag_decisions_total[5m])) * 60",
                "{{decider}} {{outcome}}",
            )
        ],
    )
    g.chart(
        "Decider cost per hour (USD)",
        [("sum(increase(rag_decider_cost_usd_total[1h]))", "Jev $/h")],
        "currencyUSD",
    )

    g.row("Drift & quality: are the questions or the answers changing?")
    g.chart("Nightly RAGAS faithfulness", [("rag_eval_faithfulness", "faithfulness")], "percentunit",
            6, "stat", [{"color": "red", "value": None}, {"color": "green", "value": 0.8}])  # fmt: skip
    g.chart("Drift score by test", [("rag_drift_score", "{{test}} {{feature}}")], "short", 12)
    g.chart("Drift alerts", [("sum(rag_drift_alert)", "tests firing")], "short", 6, "stat",
            green_red + [{"color": "red", "value": 1}])  # fmt: skip

    return {
        "uid": "legal-rag", "title": "legal-rag: service, resources, RAG, drift",
        "schemaVersion": 39, "version": 1, "refresh": "30s", "time": {"from": "now-1h", "to": "now"},
        "tags": ["legal-rag"], "panels": g.panels,
        "templating": {"list": [{
            "name": "backend", "type": "query", "datasource": PROM, "label": "LLM backend",
            "query": {"query": "label_values(rag_llm_tokens_total, backend)", "refId": "backend"},
            "definition": "label_values(rag_llm_tokens_total, backend)",
            "includeAll": True, "multi": True, "allValue": ".*", "refresh": 2,
            "current": {"text": "All", "value": "$__all"},
        }]},
        "annotations": {"list": [{
            # a restarted API process = a deploy (or a crash): drawn as a line on every chart
            "name": "Deploys / restarts", "datasource": PROM, "enable": True, "iconColor": "orange",
            "expr": 'changes(process_start_time_seconds{job="api"}[2m]) > 0',
            "step": "60s", "titleFormat": "API (re)started",
        }]},
    }  # fmt: skip


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8", newline="\n")
