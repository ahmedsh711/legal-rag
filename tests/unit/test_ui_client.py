"""The Streamlit UI's API client: SSE parsing, answer state, headers."""

import json

import httpx
import pytest
from rag_client import Answer, RagClient, is_arabic, parse_sse


def sse(*events: dict) -> str:
    return "".join(f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events)


def test_parse_sse_reads_data_lines_and_skips_the_rest():
    lines = ['data: {"type": "token", "text": "a"}', "", ": keep-alive", 'data: {"type": "done"}']
    assert [e["type"] for e in parse_sse(lines)] == ["token", "done"]


def test_answer_collects_tokens_and_the_final_event():
    answer = Answer()
    for event in [
        {"type": "token", "text": "Majority is "},
        {"type": "token", "text": "21 [Art. 44]."},
        {"type": "done", "refused": False, "sources": [{"article_number": 44}], "request_id": "r1"},
    ]:
        answer.apply(event)
    assert answer.text == "Majority is 21 [Art. 44]." and answer.done
    assert answer.sources == [{"article_number": 44}] and answer.request_id == "r1"


def test_a_blocked_answer_is_replaced_by_the_refusal():
    answer = Answer()
    answer.apply({"type": "token", "text": "an uncited claim"})
    answer.apply(
        {"type": "done", "refused": True, "replace_with": "I cannot find...", "sources": []}
    )
    assert answer.text == "I cannot find..." and answer.refused


def test_an_error_event_is_kept_for_the_user():
    answer = Answer()
    answer.apply({"type": "error", "detail": "The language model is unavailable, please retry."})
    assert answer.error.startswith("The language model") and answer.done


@pytest.mark.parametrize(
    ("text", "arabic"),
    [
        ("ما هي مدة التقادم؟", True),
        ("What is a lease?", False),
        ("ما معنى Article 44؟", True),
        ("", False),
    ],
)
def test_is_arabic(text, arabic):
    assert is_arabic(text) is arabic


def test_client_streams_events_with_its_key_and_returns_metadata():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        if request.url.path == "/metadata":
            return httpx.Response(200, json={"llm_model": "m", "prompt_version": "v3"})
        assert request.url.params["stream"] == "true"
        body = sse(
            {"type": "token", "text": "hi"}, {"type": "done", "refused": False, "sources": []}
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = RagClient("http://api", api_key="k1", transport=httpx.MockTransport(handler))
    events = list(client.stream("What is a lease?"))
    assert [e["type"] for e in events] == ["token", "done"]
    assert seen["x-api-key"] == "k1" and "ngrok-skip-browser-warning" in seen
    assert client.metadata()["prompt_version"] == "v3"


def test_client_reports_http_errors_as_an_error_event():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"detail": "Too many requests"}, headers={"Retry-After": "5"}
        )

    client = RagClient("http://api", transport=httpx.MockTransport(handler))
    [event] = list(client.stream("What is a lease?"))
    assert event["type"] == "error" and "Too many requests" in event["detail"]
