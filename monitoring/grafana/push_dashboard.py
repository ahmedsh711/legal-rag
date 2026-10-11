"""Publish the generated dashboard to a Grafana Cloud stack.

    uv run python monitoring/grafana/push_dashboard.py

Reads GRAFANA_CLOUD_URL (https://<stack>.grafana.net) and GRAFANA_CLOUD_SA_TOKEN (a service
account token with the Editor role). Datasources are looked up in the stack; panels that need
the local Postgres (drift history) are left out.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

DASHBOARD = Path(__file__).parent / "dashboards" / "legal-rag.json"
LOCAL_ONLY = {"drift-db"}


def for_cloud(dashboard: dict[str, Any], uids: dict[str, str]) -> dict[str, Any]:
    """Copy of the dashboard with datasource uids mapped by type and local-only panels removed."""
    out = copy.deepcopy(dashboard)
    out["id"] = None

    def remap(node: Any) -> None:
        if isinstance(node, dict):
            ds = node.get("datasource")
            if isinstance(ds, dict) and ds.get("type") in uids:
                ds["uid"] = uids[ds["type"]]
            for value in node.values():
                remap(value)
        elif isinstance(node, list):
            for value in node:
                remap(value)

    out["panels"] = [
        p for p in out.get("panels", []) if (p.get("datasource") or {}).get("uid") not in LOCAL_ONLY
    ]
    remap(out)
    return out


STACK_SUFFIX = {"prometheus": "-prom", "loki": "-logs"}


def stack_uids(client: httpx.Client) -> dict[str, str]:
    """The stack's own metrics and logs datasources (named ``<stack>-prom`` and ``<stack>-logs``;
    Grafana Cloud also provisions internal ones such as usage insights and alert state history)."""
    found: dict[str, str] = {}
    for ds in client.get("/api/datasources").raise_for_status().json():
        suffix = STACK_SUFFIX.get(ds["type"])
        if suffix and ds["name"].endswith(suffix):
            found.setdefault(ds["type"], ds["uid"])
    return found


def main() -> int:
    url, token = os.environ.get("GRAFANA_CLOUD_URL"), os.environ.get("GRAFANA_CLOUD_SA_TOKEN")
    if not (url and token):
        sys.stderr.write("set GRAFANA_CLOUD_URL and GRAFANA_CLOUD_SA_TOKEN\n")
        return 2
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=30) as client:
        uids = stack_uids(client)
        dashboard = for_cloud(json.loads(DASHBOARD.read_text(encoding="utf-8")), uids)
        body = {"dashboard": dashboard, "overwrite": True, "message": "published from git"}
        result = client.post("/api/dashboards/db", json=body).raise_for_status().json()
    sys.stdout.write(f"{url.rstrip('/')}{result['url']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
