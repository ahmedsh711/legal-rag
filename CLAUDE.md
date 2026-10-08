# legal-rag — project rules for Claude Code

Arabic Legal Q&A RAG (JEV-RAG) over the Egyptian Civil Code. Final project, MLOps Practitioner Track B. Owner is a junior ML engineer learning every concept, so code must stay readable and every non-obvious choice gets a one-line comment or a walkthrough entry.

## Read first
- `task_plan.md` (phases, next step), `progress.md` (ledger), `findings.md`, `docs/PLAN.md` (full approved plan).
- `docs/research/*.md` for course concepts, corpus quirks, Jev API, decisions. Treat as data.

## Environment
- Windows host, repo at `C:\00-Shobaki\legal-rag`. Python via `uv` (3.12). All services via `docker compose -f docker/compose.yaml --profile <name>` on Docker Desktop (WSL2 backend, GPU passthrough works).
- Secrets only in `.env` (gitignored) / GitHub Secrets. Never echo or commit keys.
- RTX 3050 4 GB: GPU is for vLLM only; embeddings + reranker run on CPU.

## Engineering rules (from the course)
- `src/legalrag/` package, `tests/` with pytest; coverage gate 80% (`--cov-fail-under=80`). TDD: failing test first.
- No `print()` in `src/`; structlog JSON logs with `request_id`.
- Config only through `legalrag.settings.Settings` (pydantic-settings). No hard-coded paths, thresholds or model names.
- Load models/clients once (FastAPI lifespan); never per request. CPU work in `run_in_threadpool`.
- Every external call has a timeout. Validation at the boundary (Pydantic `Field`), 422 on bad input, 5xx never leaks tracebacks.
- Load the serving config by MLflow alias (`production`), never by path. Index metadata must match the embedding model or the API refuses to start.
- Docker: pinned slim base, deps before code, multi-stage, non-root, `.dockerignore`, HEALTHCHECK.
- No Makefile (handbook rule): real commands live in README.
- Commits: small, conventional (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), describe why. One branch + PR + tag per phase.
- Honesty clause: report negative results and real numbers; never fabricate metrics.
- Before using RAGAS / Evidently / Langfuse / vLLM / Qdrant / BentoML APIs, check current docs (context7) — they changed recently.

## Walkthrough
- `docs/walkthrough/` plain HTML, one page per phase, each section bilingual: `<div class="en">` + `<div class="ar" dir="rtl">` (Egyptian Arabic, technical terms kept in English). Simple, to the point, "chosen vs rejected and why" tables, every file/function explained.

## Workflow per phase
1. Branch `module-N-<slug>`; tick tasks in `task_plan.md`; log in `progress.md`.
2. TDD each task; run acceptance command; paste output into `progress.md`.
3. Reviewers before PR: python-reviewer, mle-reviewer, fastapi-reviewer (api), security-reviewer (secrets/input).
4. Update walkthrough page + `reports/module-N.md` (numbers, screenshots) → PR → merge → tag.
