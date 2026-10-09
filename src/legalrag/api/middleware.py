"""Request-id middleware: one id per request, in every log line, the response header and (later)
the Langfuse trace. An incoming ``X-Request-ID`` is reused so ids can cross service boundaries.

It is also the last line of defence: an unexpected exception becomes a clean 500 that carries the
request id (so the user can report it) but never the traceback (that stays in the logs)."""

from __future__ import annotations

import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from legalrag.api.schemas import REQUEST_ID_PATTERN
from legalrag.logging_conf import get_logger, request_id_var

log = get_logger("legalrag.api")
_SAFE_ID = re.compile(REQUEST_ID_PATTERN)  # never trust a header blindly: it ends up in logs


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _SAFE_ID.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                log.exception("unhandled_error", method=request.method, path=request.url.path)
                response = JSONResponse(
                    {"detail": "Internal server error", "request_id": request_id}, status_code=500
                )
            # for a stream this is the time to the first byte, not the whole answer
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
            log.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=elapsed_ms,
            )
            return response
        finally:
            request_id_var.reset(token)
