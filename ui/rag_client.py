"""Small HTTP client for the legal-rag API, used by the Streamlit app."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

FIRST_LETTER = re.compile(r"[A-Za-z؀-ۿ]")


def is_arabic(text: str) -> bool:
    """Right to left if the first letter is Arabic (the "first strong character" rule)."""
    first = FIRST_LETTER.search(text)
    return bool(first) and "؀" <= first.group() <= "ۿ"


def parse_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        if line.startswith("data:"):
            yield json.loads(line[5:].strip())


@dataclass
class Answer:
    """The state of one streamed answer, built event by event."""

    text: str = ""
    done: bool = False
    refused: bool = False
    error: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    request_id: str | None = None

    def apply(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "token":
            self.text += event.get("text", "")
        elif kind == "done":
            # the API sends the refusal when it blocks an answer that was already streamed
            self.text = event.get("replace_with", self.text)
            self.refused = event.get("refused", False)
            self.sources = event.get("sources", [])
            self.guardrails = event.get("guardrails", [])
            self.timings_ms = event.get("timings_ms", {})
            self.request_id = event.get("request_id")
            self.done = True
        elif kind == "error":
            self.error = event.get("detail", "Something went wrong.")
            self.request_id = event.get("request_id")
            self.done = True


class RagClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120,
        transport: httpx.BaseTransport | None = None,
    ):
        headers = {
            "ngrok-skip-browser-warning": "1"
        }  # the tunnel's free tier shows a page otherwise
        if api_key:
            headers["X-API-Key"] = api_key
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    def health(self) -> dict[str, Any]:
        return self.http.get("/health").json()

    def metadata(self) -> dict[str, Any]:
        return self.http.get("/metadata").json()

    def stream(self, question: str, book: str | None = None) -> Iterator[dict[str, Any]]:
        body = {"question": question, "book": book}
        try:
            with self.http.stream("POST", "/ask", params={"stream": "true"}, json=body) as r:
                if r.status_code != 200:
                    r.read()
                    yield {"type": "error", "detail": _detail(r)}
                    return
                yield from parse_sse(r.iter_lines())
        except httpx.HTTPError as exc:
            yield {"type": "error", "detail": f"The API is not reachable ({type(exc).__name__})."}

    def feedback(self, request_id: str, rating: str, comment: str | None = None) -> bool:
        body = {"request_id": request_id, "rating": rating, "comment": comment}
        return self.http.post("/feedback", json=body).status_code == 202


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    wait = response.headers.get("retry-after")
    return f"{detail} (retry in {wait} s)" if wait else str(detail)
