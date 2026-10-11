"""BentoML service around the same RAG pipeline the FastAPI app serves.

Serve it from the repository root (settings come from the environment and .env, like the API):

    uv run --group bento bentoml serve serving/bento/service.py:LegalRag --port 3005
    curl -X POST http://127.0.0.1:3005/ask -H "Content-Type: application/json" \
        -d '{"question": "What is the prescription period of obligations?"}'

Build a bento (requirements.txt is exported from uv.lock; re-export it after changing deps):

    uv export --frozen --no-dev --group bento --no-hashes --no-emit-project --no-annotate \
        -o serving/bento/requirements.txt
    uv run --group bento bentoml build -f serving/bento/bentofile.yaml

Only /ask is served here. Rate limits, API keys, metrics and prediction events stay in the
FastAPI app, which remains the public entry point.
"""

from __future__ import annotations

import uuid
from http import HTTPStatus

import bentoml
from bentoml.exceptions import BentoMLException
from openai import APIConnectionError, APIStatusError
from pydantic import ConfigDict

from legalrag.api.main import ask_response, build_components, close_components, llm_failure
from legalrag.api.schemas import AskRequest, AskResponse
from legalrag.config_registry import apply_registry_config
from legalrag.guardrails import redact_pii
from legalrag.logging_conf import configure_logging, request_id_var
from legalrag.settings import get_settings


class AskInput(AskRequest):
    # same fields and validators; BentoML's OpenAPI schema takes one "example", not "examples"
    model_config = ConfigDict(
        json_schema_extra={"example": {"question": "ما هي مدة تقادم الالتزام؟"}}
    )


@bentoml.service
class LegalRag:
    """Egyptian Civil Code Q&A; every answer cites the articles it comes from."""

    def __init__(self) -> None:
        settings = get_settings()
        configure_logging(settings.log_level)
        served, _source = apply_registry_config(settings)
        self.components = build_components(served)  # once per worker, never per request

    @bentoml.api(input_spec=AskInput)
    async def ask(self, question: str, book: str | None = None) -> AskResponse:
        """Answer a question in Arabic or English with article citations."""
        pipeline = self.components.pipeline
        request_id = uuid.uuid4().hex
        token = request_id_var.set(request_id)
        try:
            question_seen = {"question": redact_pii(question)[0]}
            with pipeline.tracer.trace(request_id, "ask", input=question_seen):
                result = await pipeline.ask(question, book=book)
        except (APIConnectionError, APIStatusError) as exc:
            status, detail, _headers = llm_failure(exc)
            raise BentoMLException(detail, error_code=HTTPStatus(status)) from exc
        finally:
            request_id_var.reset(token)
        return ask_response(result, request_id)

    @bentoml.on_shutdown
    async def close(self) -> None:
        await close_components(self.components)
