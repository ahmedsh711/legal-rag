"""FastAPI service: /ask (JSON or SSE stream), /health, /live, /metadata, /feedback.

Everything heavy (Qdrant client, bge-m3, LLM client) is built once in the lifespan and kept on
``app.state``; requests only use it. ``create_app(build=...)`` lets tests swap in fakes.

    uv run uvicorn legalrag.api.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIConnectionError, APIStatusError
from prometheus_client import CollectorRegistry

from legalrag import __version__
from legalrag.api.middleware import RequestIdMiddleware
from legalrag.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorBody,
    FeedbackRequest,
    HealthResponse,
    Source,
)
from legalrag.config_registry import apply_registry_config
from legalrag.decider.base import Decider
from legalrag.decider.jev import JevDecider
from legalrag.decider.local import LocalDecider
from legalrag.generation import PROMPT_VERSION, Generator, make_client
from legalrag.guardrails import redact_pii
from legalrag.index.batcher import QueryBatcher
from legalrag.index.store import alias_target, read_metadata
from legalrag.ingest.normalize import NORMALIZATION_VERSION
from legalrag.logging_conf import configure_logging, get_logger, request_id_var
from legalrag.monitoring.events import EventLog, event_from_answer, event_from_stream_done
from legalrag.observability.metrics import RagMetrics, render
from legalrag.pipeline import RagPipeline
from legalrag.ratelimit import TokenBucket, build_limiter, client_key, known_key_hashes
from legalrag.retrieval import Retriever
from legalrag.settings import Settings, get_settings

log = get_logger(__name__)


class IndexMismatchError(RuntimeError):
    """The index was built with a different embedding model or normalization than the API uses."""


@dataclass
class Components:
    pipeline: Any  # RagPipeline (or a fake in tests)
    qdrant: Any
    alias: str
    collection: str
    index_meta: dict[str, Any]
    limiter: TokenBucket | None = None  # None = no rate limit


def check_index_compatible(meta: dict[str, Any] | None, settings: Settings) -> None:
    """Refuse to serve if questions would be embedded differently from the documents.

    The instructor's incident: a nightly rebuild changed the embedding model, the app kept the old
    one, and 18% of Arabic answers silently became "not enough information"."""
    if not meta:
        raise IndexMismatchError("index has no metadata; rebuild it with `dvc repro`")
    if meta.get("embedding_model") != settings.embedding_model:
        raise IndexMismatchError(
            f"index built with {meta.get('embedding_model')!r}, API uses {settings.embedding_model!r}"
        )
    if meta.get("embedding_revision") not in (None, settings.embedding_revision):
        raise IndexMismatchError(
            f"index built with revision {meta.get('embedding_revision')!r}, "
            f"API uses {settings.embedding_revision!r}"
        )
    if meta.get("normalization_version") not in (None, NORMALIZATION_VERSION):
        raise IndexMismatchError(
            f"index normalization {meta.get('normalization_version')!r} != {NORMALIZATION_VERSION!r}"
        )


def build_components(settings: Settings) -> Components:
    from qdrant_client import QdrantClient

    from legalrag.index.embedder import BgeM3Embedder

    client = QdrantClient(url=settings.qdrant_url, timeout=10)
    collection = alias_target(client, settings.qdrant_collection_alias)
    if collection is None:
        raise IndexMismatchError(
            f"alias {settings.qdrant_collection_alias!r} not found; build the index first"
        )
    meta = read_metadata(client, collection)
    check_index_compatible(meta, settings)
    embedder = BgeM3Embedder(
        settings.embedding_model,
        settings.embedding_device,
        settings.query_max_length,
        revision=settings.embedding_revision,
    )
    if settings.query_batch_max > 1:
        embedder = QueryBatcher(embedder, max_batch=settings.query_batch_max)
    retriever = Retriever(
        client,
        settings.qdrant_collection_alias,
        embedder,
        top_n=settings.retrieve_top_n,
        mode=settings.retrieval_mode,
    )
    generator = Generator(
        make_client(
            settings.active_llm_base_url, settings.active_llm_api_key, settings.llm_timeout_s
        ),
        settings.active_llm_model,
        settings.llm_max_tokens,
        settings.llm_temperature,
    )
    pipeline = RagPipeline(
        retriever,
        generator,
        context_size=settings.rerank_keep_top,
        top_n=settings.retrieve_top_n,
        decider=build_decider(settings),
        gate_threshold=settings.gate_threshold,
    )
    limiter = build_limiter(
        settings.redis_url, settings.rate_limit_per_minute, settings.rate_limit_burst
    )
    return Components(
        pipeline, client, settings.qdrant_collection_alias, collection, meta or {}, limiter
    )


def build_decider(settings: Settings) -> Decider | None:
    if settings.decider_backend == "jev":
        return JevDecider(
            httpx.AsyncClient(timeout=settings.jev_timeout_s),
            settings.jev_base_url,
            settings.openrouter_api_key.get_secret_value(),
            settings.jev_model,
        )
    if settings.decider_backend == "local":
        return LocalDecider.from_pretrained(
            settings.reranker_model, settings.reranker_revision, settings.embedding_device
        )
    return None


def _llm_failure(exc: Exception) -> tuple[int, str, dict[str, str]]:
    """Provider outage / overload -> 503 + Retry-After (try again later).
    Provider rejecting our request (bad key, unknown model) -> 502: our config, retrying won't help."""
    code = getattr(exc, "status_code", None)
    log.error("llm_error", error=type(exc).__name__, provider_status=code)
    if isinstance(exc, APIStatusError) and code != 429 and code < 500:
        return 502, "The language model rejected the request.", {}
    return 503, "The language model is unavailable, please retry.", {"Retry-After": "10"}


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _close(comp: Components) -> None:
    """Release connections on shutdown (test fakes have nothing to close). One failing close
    must not leave the others open."""
    llm = getattr(getattr(comp.pipeline, "generator", None), "client", None)
    decider = getattr(comp.pipeline, "decider", None)  # Jev holds an HTTP client
    retriever = getattr(comp.pipeline, "retriever", None)  # its QueryBatcher owns a thread
    closers = [
        ("qdrant", getattr(comp.qdrant, "close", None)),
        ("llm", getattr(llm, "close", None)),
        ("decider", getattr(decider, "aclose", None)),
        ("limiter", comp.limiter.aclose if comp.limiter is not None else None),
        ("embedder", getattr(getattr(retriever, "embedder", None), "close", None)),
    ]
    for name, close in closers:
        if close is None:
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 - shutdown: log and keep closing the rest
            log.exception("close_failed", resource=name)


class RateLimitedError(Exception):
    def __init__(self, retry_after_s: int):
        super().__init__(f"rate limited, retry after {retry_after_s} s")
        self.retry_after_s = retry_after_s


async def rate_limit(request: Request, response: Response) -> None:
    """Dependency for the costly endpoints: take one token for this client or answer 429.
    A dependency runs before the body is validated, so a flood of bad requests is limited too."""
    comp: Components = request.app.state.components
    if comp.limiter is None:
        return
    ip = request.client.host if request.client else None
    key = client_key(request.headers.get("X-API-Key"), ip, request.app.state.known_keys)
    verdict = await comp.limiter.take(key)
    outcome = "limited" if not verdict.allowed else "degraded" if verdict.degraded else "allowed"
    request.app.state.metrics.ratelimit.labels(outcome).inc()
    if not verdict.allowed:
        log.warning("rate_limited", retry_after_s=verdict.retry_after_s)
        raise RateLimitedError(verdict.retry_after_s)
    if not verdict.degraded:  # an honest client can slow down before it ever sees a 429
        response.headers["X-RateLimit-Remaining"] = str(verdict.remaining)


RATE_LIMITED = {
    429: {
        "model": ErrorBody,
        "description": "Too many requests from this client; retry after Retry-After seconds",
        "headers": {
            "Retry-After": {
                "description": "seconds until a request is allowed again",
                "schema": {"type": "integer"},
            }
        },  # fmt: skip
    }
}


def _source(c: Any) -> Source:
    return Source(
        article_number=c.article_number,
        citation=c.citation,
        citation_ar=c.citation_ar,
        is_repealed=c.is_repealed,
        quality_flags=list(c.quality_flags),
    )


def _append_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_app(
    build: Callable[[Settings], Components] = build_components, settings: Settings | None = None
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        # runtime knobs from models:/legal-rag-config@production when CONFIG_SOURCE=mlflow
        served, app.state.config_source = apply_registry_config(settings)
        app.state.settings = served
        app.state.components = build(served)  # load once, never per request
        log.info("startup", collection=app.state.components.collection,
                 llm=served.active_llm_model, config=app.state.config_source)  # fmt: skip
        app.state.serving = {
            "prompt_version": PROMPT_VERSION, "llm_model": served.active_llm_model,
            "decider": served.decider_backend,
            "index_collection": app.state.components.collection,
        }  # fmt: skip
        app.state.metrics.set_info(app_version=__version__,
                                   config_source=app.state.config_source,
                                   **app.state.serving)  # fmt: skip
        try:
            yield
        finally:
            await _close(app.state.components)

    app = FastAPI(
        title="legal-rag: Egyptian Civil Code Q&A",
        version=__version__,
        description="Ask in Arabic or English; answers cite the articles they come from.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.state.known_keys = known_key_hashes(settings.api_keys.get_secret_value())
    # one registry per app: tests build many apps in one process; production builds one
    app.state.metrics_registry = CollectorRegistry()
    app.state.metrics = RagMetrics(app.state.metrics_registry)
    app.state.events = EventLog(settings.events_dir) if settings.events_dir else None

    async def record(event_factory: Callable[[], Any]) -> None:
        """Write one prediction event in a worker thread; a failed write never fails an answer."""
        if app.state.events is None:
            return
        try:
            await asyncio.to_thread(app.state.events.write, event_factory())
        except Exception:  # noqa: BLE001 - monitoring must not take answers down
            log.exception("event_write_failed")

    @app.exception_handler(RateLimitedError)
    async def _too_many(request: Request, exc: RateLimitedError) -> JSONResponse:
        return JSONResponse(
            {
                "detail": "Too many requests, please retry later.",
                "request_id": request_id_var.get(),
            },
            status_code=429,
            headers={"Retry-After": str(exc.retry_after_s)},
        )

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]}
            for e in exc.errors()
        ]
        return JSONResponse({"detail": errors, "request_id": request_id_var.get()}, status_code=422)

    @app.exception_handler(APIConnectionError)  # includes APITimeoutError
    @app.exception_handler(APIStatusError)
    async def _llm_error(request: Request, exc: Exception) -> JSONResponse:
        status, detail, headers = _llm_failure(exc)
        return JSONResponse(
            {"detail": detail, "request_id": request_id_var.get()}, status, headers=headers
        )

    @app.get("/live", tags=["ops"])
    async def live() -> dict[str, str]:
        """Liveness: the process answers. Kubernetes restarts the pod if this fails."""
        return {"status": "alive"}

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(request: Request) -> Any:
        """Readiness: the index is reachable and not empty. Load balancers send traffic only then."""
        comp: Components = request.app.state.components
        try:  # sync client call in a thread: a slow Qdrant must not freeze every other request
            points = (await asyncio.to_thread(comp.qdrant.count, comp.alias)).count
        except Exception as exc:  # noqa: BLE001 - any backend failure means "not ready"
            log.error("health_check_failed", error=type(exc).__name__)
            return JSONResponse(
                {"status": "unhealthy", "reason": "index unreachable"}, status_code=503
            )
        if points == 0:
            return JSONResponse({"status": "unhealthy", "reason": "index empty"}, status_code=503)
        return HealthResponse(
            status="healthy",
            documents_indexed=int(comp.index_meta.get("articles", 0)),
            points=points,
            index_collection=comp.collection,
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request) -> Response:
        """Prometheus scrape endpoint (text format). Only reachable inside the compose network
        and on 127.0.0.1; a public deployment would put it behind the proxy's allow-list."""
        body, content_type = render(request.app.state.metrics_registry)
        return Response(body, media_type=content_type)

    @app.get("/metadata", tags=["ops"])
    async def metadata(request: Request) -> dict[str, Any]:
        """What is serving right now: versions of the app, prompt, models and index."""
        comp: Components = request.app.state.components
        served: Settings = request.app.state.settings
        return {
            "app_version": __version__,
            "prompt_version": PROMPT_VERSION,
            "config_source": request.app.state.config_source,
            "llm_backend": served.llm_backend,
            "llm_model": served.active_llm_model,
            "embedding_model": served.embedding_model,
            "embedding_revision": served.embedding_revision,
            "decider_backend": served.decider_backend,
            "retrieval_mode": served.retrieval_mode,
            "gate_threshold": served.gate_threshold,
            "index": {"alias": comp.alias, "collection": comp.collection, **comp.index_meta},
        }

    @app.post("/ask", response_model=AskResponse, tags=["qa"],
              dependencies=[Depends(rate_limit)], responses=RATE_LIMITED)  # fmt: skip
    async def ask(request: Request, body: AskRequest, stream: bool = Query(False)) -> Any:
        """Answer a question about the Egyptian Civil Code with article citations.
        `?stream=true` sends the answer token by token as Server-Sent Events."""
        pipeline = request.app.state.components.pipeline
        metrics: RagMetrics = request.app.state.metrics
        backend = request.app.state.settings.llm_backend
        request_id = request_id_var.get()
        if stream:

            async def events() -> AsyncIterator[str]:
                try:
                    async for event in pipeline.ask_stream(body.question, book=body.book):
                        if event["type"] == "done":
                            event["request_id"] = request_id
                            metrics.observe_stream_done(event, backend)
                            await record(
                                lambda e=event: event_from_stream_done(
                                    e, body.question, request_id, request.app.state.serving
                                )
                            )
                        yield _sse(event)
                except (APIConnectionError, APIStatusError) as exc:
                    # the 200 status line is already sent; tell the client in the stream itself
                    yield _sse({"type": "error", "detail": _llm_failure(exc)[1],
                                "request_id": request_id})  # fmt: skip
                except Exception:
                    log.exception("stream_failed")
                    yield _sse({"type": "error", "detail": "Internal server error",
                                "request_id": request_id})  # fmt: skip

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                # no-cache + no proxy buffering, or tokens arrive all at once at the end
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        result = await pipeline.ask(body.question, book=body.book)
        metrics.observe_answer(result, backend)
        await record(lambda: event_from_answer(result, body.question, request_id, "ask",
                                               request.app.state.serving))  # fmt: skip
        return AskResponse(
            answer=result.answer,
            language=result.language,
            sources=[_source(c) for c in result.sources],
            refused=result.refused,
            invalid_citations=result.invalid_citations,
            request_id=request_id,
            prompt_version=result.prompt_version,
            guardrails=result.guardrails,
            model=result.model,
            usage=result.usage,
            timings_ms=result.timings_ms,
        )

    @app.post("/feedback", status_code=202, tags=["qa"],
              dependencies=[Depends(rate_limit)], responses=RATE_LIMITED)  # fmt: skip
    async def feedback(body: FeedbackRequest) -> dict[str, str]:
        """Thumbs up/down on an answer, keyed by its request id (becomes a Langfuse score later).
        Same rate limit as /ask (an unlimited write endpoint fills the disk), and the comment is
        stored with PII redacted, like questions."""
        record = {
            "at": datetime.now(UTC).isoformat(),
            "request_id": body.request_id,
            "rating": body.rating,
            "comment": redact_pii(body.comment)[0] if body.comment else None,
        }
        await asyncio.to_thread(_append_line, Path(settings.feedback_path), record)
        return {"status": "recorded"}

    return app


app = create_app()  # what `uvicorn legalrag.api.main:app` serves
