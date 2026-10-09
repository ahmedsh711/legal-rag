# legal-rag — Arabic Legal Q&A (JEV-RAG) over the Egyptian Civil Code

Final project for the ITI × MLOps MENA **MLOps Practitioner** course, Track B (LLM / RAG).
Ask a question about the Egyptian Civil Code in Arabic or English and get an answer **with article citations**. The pipeline's *deciding* steps (routing, reranking, "can we answer?", claim checking) run on the Jev decision model; only the *writing* step runs on an LLM.

> Status: Phase 2. Vanilla RAG (hybrid retrieval + cited answers) behind a FastAPI service in Docker. The Jev decision steps arrive in Phases 3–4.

## Quickstart (3 commands)

Prerequisites: Docker Desktop, [uv](https://docs.astral.sh/uv/), `cp .env.example .env` with your `OPENROUTER_API_KEY`, and the course PDF at `data/raw/egyptian_civil_code.pdf` (or `uv run dvc pull` if you can reach the DVC remote).

```bash
docker compose --env-file .env -f docker/compose.yaml --profile core up -d   # 1. qdrant + redis + api
uv run dvc repro                                                             # 2. PDF -> articles.json -> validate -> Qdrant index
curl -s localhost:8000/ask -H 'content-type: application/json'      -d '{"question":"ما هي مدة تقادم الالتزام؟"}'                             # 3. ask (add ?stream=true for SSE)
```

Step 2 embeds 2,241 texts with bge-m3 on CPU (~17 min on a laptop, once; DVC skips it when nothing changed). The API container waits for the index: until the alias exists it exits and Docker restarts it. Check with `curl localhost:8000/health` (`"status": "ok"` once ready) and see what is serving with `curl localhost:8000/metadata`.

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{"question": "...", "book": null}` → answer, cited sources, refusal flag, request id, token usage, timings. `?stream=true` → Server-Sent Events |
| `POST /feedback` | `{"request_id": "...", "rating": "up" \| "down", "comment": "..."}` |
| `GET /health` | readiness: 200 only when the index is reachable and not empty |
| `GET /live` | liveness: the process answers |
| `GET /metadata` | app, prompt, LLM, embedding model and index versions |

Interactive docs: http://localhost:8000/docs

## Developer commands

There is no Makefile on purpose: the course handbook asks for the real commands in the README.

```bash
uv sync                                                              # install (.venv, Python 3.12)
uv run ruff check src tests && uv run ruff format --check src tests  # lint + format check
uv run pytest                                                        # tests; 80% coverage gate is in pyproject.toml
uv run pre-commit install                                            # git hooks
docker compose -f docker/compose.yaml --profile core --profile tracking up -d
```

## Repository map

```
src/legalrag/        installable package: settings, ingest, index, retrieval, decider, api, eval, monitoring
tests/               pytest suite
docker/              compose.yaml (profiles: core, tracking, llm, monitoring, llm-observability, airflow, serving)
docs/walkthrough/    bilingual (EN / Egyptian Arabic) learning guide, one page per phase
docs/research/       research digests the design was built from
docs/PLAN.md         approved plan; task_plan.md / progress.md / findings.md track execution
reports/             one lab report per module with measured numbers and screenshots
```

## Learning guide

Open `docs/walkthrough/index.html` in a browser. Every section has an EN / AR / Both toggle.

## Changelog by session

- Session 1 (packaging, API, Docker): in progress.
