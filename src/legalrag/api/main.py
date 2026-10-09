"""FastAPI service: /ask (JSON or SSE stream), /health, /live, /metadata, /feedback.

Everything heavy (Qdrant client, bge-m3, LLM client) is built once in the lifespan and kept on
``app.state``; requests only use it. ``create_app(build=...)`` lets tests swap in fakes.

    uv run uvicorn legalrag.api.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIConnectionError, APIStatusError

from legalrag import __version__
from legalrag.api.middleware import RequestIdMiddleware
from legalrag.api.schemas import AskRequest, AskResponse, FeedbackRequest, HealthResponse, Source
from legalrag.decider.base import Decider
from legalrag.decider.jev import JevDecider
from legalrag.decider.local import LocalDecider
from legalrag.generation import PROMPT_VERSION, Generator, make_client
from legalrag.index.store import alias_target, read_metadata
from legalrag.ingest.normalize import NORMALIZATION_VERSION
from legalrag.logging_conf import configure_logging, get_logger, request_id_var
from legalrag.pipeline import RagPipeline
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
    return Components(pipeline, client, settings.qdrant_collection_alias, collection, meta or {})


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
    """Release connections on shutdown (test fakes have nothing to close)."""
    if hasattr(comp.qdrant, "close"):
        comp.qdrant.close()
    llm = getattr(getattr(comp.pipeline, "generator", None), "client", None)
    if hasattr(llm, "close"):
        await llm.close()


def _source(c: Any) -> Source:
    return Source(
        article_number=c.article_number,
        citation=c.citation,
        citation_ar=c.citation_ar,
        is_repealed=c.is_repealed,
        quality_flags=list(c.quality_flags),
    )


def create_app(
    build: Callable[[Settings], Components] = build_components, settings: Settings | None = None
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        app.state.settings = settings
        app.state.components = build(settings)  # load once, never per request
        log.info(
            "startup", collection=app.state.components.collection, llm=settings.active_llm_model
        )
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

    @app.get("/metadata", tags=["ops"])
    async def metadata(request: Request) -> dict[str, Any]:
        """What is serving right now: versions of the app, prompt, models and index."""
        comp: Components = request.app.state.components
        return {
            "app_version": __version__,
            "prompt_version": PROMPT_VERSION,
            "llm_backend": settings.llm_backend,
            "llm_model": settings.active_llm_model,
            "embedding_model": settings.embedding_model,
            "embedding_revision": settings.embedding_revision,
            "decider_backend": settings.decider_backend,
            "index": {"alias": comp.alias, "collection": comp.collection, **comp.index_meta},
        }

    @app.post("/ask", response_model=AskResponse, tags=["qa"])
    async def ask(request: Request, body: AskRequest, stream: bool = Query(False)) -> Any:
        """Answer a question about the Egyptian Civil Code with article citations.
        `?stream=true` sends the answer token by token as Server-Sent Events."""
        pipeline = request.app.state.components.pipeline
        request_id = request_id_var.get()
        if stream:

            async def events() -> AsyncIterator[str]:
                try:
                    async for event in pipeline.ask_stream(body.question, book=body.book):
                        if event["type"] == "done":
                            event["request_id"] = request_id
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
        return AskResponse(
            answer=result.answer,
            language=result.language,
            sources=[_source(c) for c in result.sources],
            refused=result.refused,
            invalid_citations=result.invalid_citations,
            request_id=request_id,
            prompt_version=result.prompt_version,
            model=result.model,
            usage=result.usage,
            timings_ms=result.timings_ms,
        )

    @app.post("/feedback", status_code=202, tags=["qa"])
    def feedback(body: FeedbackRequest) -> dict[str, str]:
        """Thumbs up/down on an answer, keyed by its request id (becomes a Langfuse score later).
        Plain ``def``: FastAPI runs it in a worker thread, so the file write never blocks the loop."""
        path = Path(settings.feedback_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"at": datetime.now(UTC).isoformat(), **body.model_dump()}
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"status": "recorded"}

    return app


app = create_app()  # what `uvicorn legalrag.api.main:app` serves
