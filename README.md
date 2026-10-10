# legal-rag — Arabic Legal Q&A (JEV-RAG) over the Egyptian Civil Code

[![CI](https://github.com/ahmedsh711/legal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedsh711/legal-rag/actions/workflows/ci.yml)

Final project for the ITI × MLOps MENA **MLOps Practitioner** course, Track B (LLM / RAG).
Ask a question about the Egyptian Civil Code in Arabic or English and get an answer **with article citations**. The pipeline's *deciding* steps (routing, reranking, "can we answer?", claim checking) run on the Jev decision model; only the *writing* step runs on an LLM.

> Status: Phase 4. Guarded (PII redaction, injection block), rate-limited, CI-gated (lint, tests, RAGAS smoke gate, image build/push) and load-tested; generation on the Gemini API or a self-hosted vLLM on the GPU. Production config: dense retrieval + the Jev decider, served by alias from the MLflow registry. Numbers in `reports/module-3.md` and `reports/module-4.md`.

## Quickstart (3 commands)

Prerequisites: Docker Desktop, [uv](https://docs.astral.sh/uv/), `cp .env.example .env` with your `OPENROUTER_API_KEY`, and the course PDF at `data/raw/egyptian_civil_code.pdf` (or `uv run dvc pull` if you can reach the DVC remote).

```bash
docker compose --env-file .env -f docker/compose.yaml --profile core up -d   # 1. qdrant + redis + api
uv run dvc repro                                                             # 2. PDF -> articles.json -> validate -> Qdrant index
curl -s 127.0.0.1:8000/ask -H 'content-type: application/json' -d '{"question":"ما هي مدة تقادم الالتزام؟"}'   # 3. ask
```

- **Step 2** embeds 2,241 texts with bge-m3 on CPU, once (10–30 min on a laptop). The `index` stage always runs because its real output lives in Qdrant, which DVC cannot see; when the index is already there it is reused in seconds.
- **The API container** downloads bge-m3 (2.3 GB) into the `hf_cache` volume on its first start, and waits for the index: until the alias exists it exits and Docker restarts it. `curl 127.0.0.1:8000/health` returns `"status": "healthy"` once it is ready; `curl 127.0.0.1:8000/metadata` shows what is serving.
- **Streaming:** add `?stream=true` to `/ask` for Server-Sent Events (`token` events, then one `done` event with sources and timings).
- **Port 8000 taken** on your machine? Set `API_HOST_PORT=8010` in `.env` and use that port.
- Use `127.0.0.1`, not `localhost`: ports are published on IPv4 loopback only, and on Windows `localhost` tries IPv6 first (each Python call to Qdrant waited ~2 s).
- **Arabic from Git Bash on Windows:** `curl -d '...'` sends the text through the ANSI code page and the API receives it mangled. Put the JSON in a UTF-8 file and use `curl --data-binary @question.json`, or send it from Python.

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{"question": "...", "book": null}` → answer, cited sources, refusal flag, request id, token usage, per-stage timings, `guardrails` (which guards fired). `?stream=true` → Server-Sent Events. Rate-limited: `429` + `Retry-After` |
| `POST /feedback` | `{"request_id": "...", "rating": "up" \| "down", "comment": "..."}` (comment stored with PII redacted; rate-limited) |
| `GET /health` | readiness: 200 only when the index is reachable and not empty |
| `GET /live` | liveness: the process answers |
| `GET /metadata` | app, prompt, LLM, embedding model, index, decider and where the config came from |

Interactive docs: http://127.0.0.1:8000/docs

## Evaluation and experiments

Start the tracking stack (Postgres + MinIO + MLflow at http://127.0.0.1:5000), then run experiments. Each command is one MLflow run with the config as params, metrics per language, and lineage tags (git SHA, `articles.json` md5, Qdrant collection, golden-set md5).

```bash
docker compose --env-file .env -f docker/compose.yaml --profile core --profile tracking up -d
uv sync --group eval                                                       # RAGAS + MLflow client
uv run python -m legalrag.eval.run --name ret-dense-jev --decider jev --mode dense --retrieval-only --mlflow   # ranking + gate, no LLM
uv run python -m legalrag.eval.run --name e2e-dense-jev --decider jev --mode dense --gate 0.5 --ragas --mlflow # full pipeline + RAGAS judge
uv run python -m legalrag.eval.track register --run-id <run id> --alias production                            # promote that run's config
uv run python -m legalrag.eval.judge calibrate --mlflow                                                         # judge vs human labels
```

With `CONFIG_SOURCE=mlflow` the API reads `models:/legal-rag-config@production` at startup; promoting or rolling back a config is moving the alias and restarting the API (no rebuild). Free-tier providers: set `LLM_RPM` / `JUDGE_RPM` so evaluations pace themselves under the per-minute limit.

## Serving, guardrails and load (Phase 4)

```bash
# self-hosted generation on an NVIDIA GPU (4 GB is enough for Qwen2.5-1.5B AWQ); the image is ~22 GB on disk
docker compose --env-file .env -f docker/compose.yaml -f docker/compose.vllm.yaml --profile core --profile llm up -d
uv run python -m legalrag.eval.llm_bench --name vllm --levels 1 4 8 16 --vram       # TTFT / tok/s per concurrency
uv run python -m legalrag.eval.guardrails_eval                                       # guard detection + false positives
uv run python -m legalrag.eval.smoke                                                 # the CI quality gate, locally

# load test: issued keys for the users, then a stepped ramp (Locust as a module: the .exe shim can hang on Windows)
export LOADTEST_API_KEYS=$(seq -f "loadtest-%g" 1 100 | paste -sd, -)
docker compose --env-file .env -f docker/compose.yaml -f docker/compose.vllm.yaml -f docker/compose.loadtest.yaml --profile core --profile llm up -d
uv run python loadtest/sample_stats.py --out reports/load/run-stats.csv --seconds 490 &
LOCUST_STEPS=5,10,20,40 LOCUST_STEP_SECONDS=120 uv run --group load python -m locust -f loadtest/locustfile.py --headless --host http://127.0.0.1:8010 --csv reports/load/run
```

- **Guards** run before retrieval: national IDs, mobile numbers and emails become `[NATIONAL_ID]` / `[PHONE]` / `[EMAIL]`; prompt-injection attempts get the normal refusal and spend no tokens. Every guard that fires is listed in the response.
- **Rate limit:** `RATE_LIMIT_PER_MINUTE` (default 30) with a burst of `RATE_LIMIT_BURST` (10) per client; a client is an issued key from `API_KEYS` (comma-separated), otherwise its IP. Redis down = served without a limit (logged).
- **CI** (`.github/workflows/ci.yml`): lint → tests → RAGAS smoke gate (needs the `GEMINI_API_KEY` secret) + image build; on `main` the image is pushed as `sha-<commit>` and `latest`.

## Developer commands

There is no Makefile on purpose: the course handbook asks for the real commands in the README.

```bash
uv sync                                                              # install (.venv, Python 3.12)
uv run ruff check src tests loadtest && uv run ruff format --check src tests loadtest  # lint + format
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

- Session 1 (packaging, API, Docker): v0.1.0, image `ahmedshobaki/legal-rag-api:0.1.0`.
- Session 2 (evaluation, MLflow, DVC lineage): golden set, exact metrics + RAGAS judge with human calibration, an MLflow run per experiment, Jev vs local reranker ablation, config registry with alias `production` (Phase 3).
- Session 3 (serving, CI/CD, load): guardrails measured as metrics, Redis token bucket, CI quality gate + image push by commit sha, vLLM on a 4 GB GPU, load test with one tuning change (+68 % answers/s) (Phase 4, v0.3.0).
