# legal-rag — Arabic Legal Q&A (JEV-RAG) over the Egyptian Civil Code

Final project for the ITI × MLOps MENA **MLOps Practitioner** course, Track B (LLM / RAG).
Ask a question about the Egyptian Civil Code in Arabic or English and get an answer **with article citations**. The pipeline's *deciding* steps (routing, reranking, "can we answer?", claim checking) run on the Jev decision model; only the *writing* step runs on an LLM.

> Status: Phase 0 (bootstrap). The 3-command quickstart is completed in Phase 2.

## Quickstart (3 commands)

```bash
docker compose -f docker/compose.yaml --profile core up -d          # 1. services
uv run python -m legalrag.index.build                               # 2. index the corpus
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"ما هي مدة التقادم في الالتزامات؟"}'             # 3. ask
```

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
