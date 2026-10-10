"""A stand-in for Slack/Discord: receives Alertmanager webhooks and records each one.

    python monitoring/alert_sink.py            # listens on :8080, appends to /data/alerts.jsonl

One JSON line per alert per delivery (channel = /page or /ticket, status firing/resolved, name,
labels, first action). Standard library only, so it runs in the plain python image.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

OUT = os.environ.get("ALERT_LOG", "/data/alerts.jsonl")


class Sink(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - name fixed by http.server
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        lines = []
        for alert in body.get("alerts", []):
            record = {
                "received": datetime.now(UTC).isoformat(),
                "channel": self.path.strip("/"),
                "status": alert.get("status"),
                "alert": alert.get("labels", {}).get("alertname"),
                "labels": alert.get("labels", {}),
                "summary": alert.get("annotations", {}).get("summary"),
                "first_action": alert.get("annotations", {}).get("first_action"),
                "starts_at": alert.get("startsAt"),
            }
            lines.append(json.dumps(record, ensure_ascii=False))
        with open(OUT, "a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))
        for line in lines:
            print(line, flush=True)  # noqa: T201 - the container log is the point
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:  # the JSON lines above are the log
        pass


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    HTTPServer(("0.0.0.0", 8080), Sink).serve_forever()  # noqa: S104 - inside the compose network
