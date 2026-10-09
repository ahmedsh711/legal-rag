# Progress Log

## Session: 2026-10-08

### Phase 0: Bootstrap + concept walkthrough

- **Status:** in_progress
- **Started:** 2026-10-08 13:00
- Actions taken:
  - Research: handbook read in full; 3 digest agents (sessions, LinkedIn+decks, corpus); web research on JEV-RAG, Jev API, Arabic RAG paper, vLLM on RTX 3050, RAGAS/Evidently/Langfuse versions
  - Plan approved (see docs/PLAN.md); repo location changed by user to C:\00-Shobaki\legal-rag
  - Created repo, .gitignore/.gitattributes, GitHub remote ahmedsh711/legal-rag; copied corpus + course materials + digests; wrote .wslconfig (16 GB)
  - Ruling: superpowers executing-plans ledger scripts not used; this file + task_plan.md are the ledger (planning-with-files), ralph-loop drives each phase — cost if wrong: none, same information
  - 2026-10-09: session resumed; installed jq 1.8.1 into ~/.local/bin (ralph-loop stop hook needs it); fixed mlflow plugin hook quoting (path with space); re-armed ralph loop
  - Scaffold: settings.py (pydantic-settings, SecretStr), logging_conf.py (structlog JSON + request_id), tests (5), ruff + pre-commit (+detect-secrets baseline), compose skeleton (core/tracking), README
  - Ruling: concept walkthrough split into index + 4 pages (00a–00d) instead of one 00-concepts.html — one 140 KB page is hard to read; same content — cost if wrong: none
  - Ruling: minimal CI (lint+test) added in Phase 0 instead of Phase 4 — gives PR #1 a real check; Phase 4 extends it with RAGAS gate + build/push — cost if wrong: none
  - Walkthrough checked: tag balance OK on all 5 pages; headless Edge screenshot of 00b (dark mode) shows aligned EN/AR columns; fixed Arabic inside <code> (32 spans) to use Arabic font
- Files created/modified:
  - task_plan.md, findings.md, progress.md, CLAUDE.md, .env.example, docs/research/*, data/raw/egyptian_civil_code.pdf

### Phase 1: Corpus pipeline

- **Status:** complete (PR #2 merged)
- Actions taken:
  - Surveyed pypdf output (header counts, reversed digits, layouts); TDD parser on 3 synthetic samples; 4 audit rounds on the real PDF
  - DVC with MinIO (bitnamilegacy image); wipe-and-pull test exposed .gitignore bug, fixed
  - Reviews: python-reviewer (block on start marker, CLI errors, long function) and mle-reviewer (approve with warnings) — all HIGH/MEDIUM fixed
  - Ruling: paragraph-marker count check declined — English translation has no paragraph numbers — cost if wrong: none (ratio check covers mis-splits)
  - Ruling: no git SHA in articles.meta.json — keeps the DVC output deterministic; MLflow runs tag git SHA — cost if wrong: low
  - Ruling: real-PDF integration tests skip in CI (DVC remote is local MinIO) — run locally before every PR — cost if wrong: medium
  - Ruling: truststore injected in legalrag/__init__.py — Python's certifi fails behind this machine's HTTPS inspection — cost if wrong: none
  - Ruling: docs-lookup agent had no Context7 access; Phase 2 APIs will be verified by installing and testing the libraries
- Files: src/legalrag/ingest/{normalize,schema,params,parse,validate}.py, tests/unit/*, tests/integration/test_corpus_real.py, params.yaml, dvc.yaml, dvc.lock, reports/module-1.md, docs/walkthrough/01-corpus.html

### Phase 2: Vanilla RAG + FastAPI + Docker

- **Status:** in_progress (branch module-2-rag-api)
- Actions taken:
  - TDD index (embedder, store, build), retrieval (hybrid RRF + explicit refs), generation, pipeline, API (schemas, middleware, main); Dockerfile + entrypoint + compose core
  - Verified qdrant-client by experiment (in-memory hybrid RRF + aliases) because docs-lookup had no Context7
  - First real index: 2,241 points, 1,028 s embedding; second (pinned revision): 3,764 s; power scheme "Silent" observed during the third run (cause of the variance not verified)
  - Disk full (C: 300 MB free): stopped the parity run that pulled the whole bge-m3 repo; removed only own caches; asked user, who chose "decide later" for Phases 4–6 disk space
  - Pinned bge-m3 revision 9a0624b8 (params + settings + index metadata + startup check)
  - DVC skipped a stage after mid-run dep edits → forced rerun; rule recorded
  - CRLF outputs on Windows → `newline="\n"` in every writer + tests; ruff LF; pre-commit mixed-line-ending
  - Docker Hub user is `ahmedshobaki` (from the Docker Desktop login), not the GitHub handle
  - Reviews (python, fastapi, security): no CRITICAL. Fixed: SSE error event, provider 4xx → 502, stream closed on disconnect + lazy request, /health off the event loop, /feedback sync + id pattern, lifespan cleanup, middleware finally/fullmatch, regex boundaries, explicit refs kept in context, empty choices, volume dirs owned by app, entrypoint -f, CA placeholder tracked, ports on 127.0.0.1, required passwords, qdrant healthcheck, .env.* ignored
  - Ruling: HEALTHCHECK stays on /health (readiness) — Docker does not restart unhealthy containers, so "unhealthy" honestly means "cannot serve"; k8s gets separate probes in Phase 6 — cost if wrong: low
  - Ruling: api keeps `env_file: ../.env` — all services are local and loopback-only; per-service secrets come with k8s Secrets in Phase 6 — cost if wrong: low
  - Ruling: /ask and /feedback throttling + auth deferred to Phase 4 (Redis token bucket) — ports are loopback-only now — cost if wrong: low
  - Ruling: pinned revision kept in IndexParams default + Settings + params.yaml — the startup check refuses any drift — cost if wrong: none
  - Ruling: Docker base images pinned by tag, not digest — course scale; Phase 4 CI builds are reproducible from uv.lock — cost if wrong: low

## Test Results

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| pytest | `uv run pytest` | 5 pass, cov >= 80% | 5 passed, 97.37% | pass |
| pre-commit | `pre-commit run --all-files` | all hooks pass | 9/9 Passed | pass |
| compose | `docker compose ... config --quiet` | valid | OK | pass |
| walkthrough tags | html.parser balance check | no unclosed tags | 5/5 OK | pass |
| Phase 1 pytest | `uv run pytest` | all pass, cov >= 80% | 84 passed, 97.20% | pass |
| dvc repro | real PDF | 0 errors | 1149 total / 1093 live / 56 repealed / 0 errors / 5 warnings | pass |
| dvc pull after wiping cache | MinIO | same hashes | pdf 5086ef5f…, articles 2 files fetched | pass |

## Error Log

| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-10-08 | Windows gh not authenticated | 1 | token from WSL gh, `gh auth login --with-token`, `gh auth setup-git` |

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 0 |
| Where am I going? | Phases 1–7 per task_plan.md |
| What's the goal? | Production JEV-RAG over Egyptian Civil Code + bilingual walkthrough |
| What have I learned? | See findings.md and docs/research/ |
| What have I done? | See above |
