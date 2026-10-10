"""Traces: the story of one request, stage by stage (Langfuse, on OpenTelemetry).

A metric says "p95 retrieve time doubled"; a trace says "this request spent 3.1 s in retrieve,
saw articles 147 and 148, the model wrote 41 tokens and they cost $0.00002". Every API request
becomes one trace whose id is derived from our request id, so a log line leads straight to it.

The pipeline only knows the small ``Tracer`` interface below:
- ``NoopTracer`` (default): nothing is sent; evaluation runs and tests stay unchanged.
- ``RecordingTracer``: keeps the observations in memory (tests, debugging).
- ``LangfuseTracer``: the real one, used by the API when Langfuse keys are configured.
What is traced: the *redacted* question, the articles, the prompt and the answer. It goes to our
own self-hosted Langfuse only; PII never gets this far (the guard runs first).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langfuse import propagate_attributes


class _NoSpan:
    def update(self, **fields: Any) -> None:
        pass


class NoopTracer:
    @contextmanager
    def trace(self, request_id: str, name: str, **fields: Any) -> Iterator[Any]:
        yield _NoSpan()

    @contextmanager
    def span(self, name: str, as_type: str = "span", **fields: Any) -> Iterator[Any]:
        yield _NoSpan()

    def score(self, name: str, value: float | str, data_type: str | None = None,
              comment: str | None = None) -> None:  # fmt: skip
        pass

    def flush(self) -> None:
        pass


class _Recorded:
    def __init__(self, record: dict[str, Any]):
        self.record = record

    def update(self, **fields: Any) -> None:
        self.record.update(fields)


class RecordingTracer(NoopTracer):
    def __init__(self) -> None:
        self.traces: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []

    @contextmanager
    def trace(self, request_id: str, name: str, **fields: Any) -> Iterator[Any]:
        record = {"request_id": request_id, "name": name, **fields}
        self.traces.append(record)
        yield _Recorded(record)

    @contextmanager
    def span(self, name: str, as_type: str = "span", **fields: Any) -> Iterator[Any]:
        record = {"name": name, "as_type": as_type, **fields}
        self.spans.append(record)
        yield _Recorded(record)

    def score(self, name: str, value: float | str, data_type: str | None = None,
              comment: str | None = None) -> None:  # fmt: skip
        self.scores.append({"name": name, "value": value, "data_type": data_type})


class LangfuseTracer(NoopTracer):
    """Thin adapter over the Langfuse v4 SDK client (``langfuse.Langfuse``)."""

    def __init__(self, client: Any):
        self.client = client

    @contextmanager
    def trace(self, request_id: str, name: str, tags: Sequence[str] = (),
              metadata: dict[str, Any] | None = None, version: str | None = None,
              input: Any = None) -> Iterator[Any]:  # noqa: A002 - Langfuse's own field name  # fmt: skip
        trace_id = self.client.create_trace_id(seed=request_id)  # request id -> trace id
        with propagate_attributes(trace_name=name, tags=list(tags), metadata=metadata or {},
                                  version=version):  # fmt: skip
            with self.client.start_as_current_observation(
                trace_context={"trace_id": trace_id}, name=name, as_type="span", input=input
            ) as root:
                yield root

    @contextmanager
    def span(self, name: str, as_type: str = "span", **fields: Any) -> Iterator[Any]:
        with self.client.start_as_current_observation(name=name, as_type=as_type, **fields) as obs:
            yield obs

    def score(self, name: str, value: float | str, data_type: str | None = None,
              comment: str | None = None) -> None:  # fmt: skip
        self.client.score_current_trace(name=name, value=value, data_type=data_type,
                                        comment=comment)  # fmt: skip

    def flush(self) -> None:
        self.client.flush()


def build_tracer(host: str, public_key: str, secret_key: str, environment: str) -> NoopTracer:
    """A Langfuse tracer when keys are configured, otherwise the no-op one."""
    if not (public_key and secret_key):
        return NoopTracer()
    from langfuse import Langfuse

    client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=host,
                      environment=environment, timeout=5)  # fmt: skip
    return LangfuseTracer(client)
