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

- **Status:** complete (PR #3 merged, tag v0.1.0, image ahmedshobaki/legal-rag-api:0.1.0 digest d46a8d58)
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

### Phase 3: Evaluation, MLflow, decider + JEV ablation

- **Status:** complete (PR #4, tag v0.2.0); judge calibration *run* carried into Phase 4 (Gemini quota)
- Results: dense+Jev MRR 1.000 vs hybrid baseline 0.878 (in-sample, 24 independent answerable pairs); held-out 5 pairs 6/6 + 4/4; Jev gate margin 0.04 vs 0.79 → gate 0.5; local cross-encoder ~9 s CPU/question; e2e faithfulness 0.969 vs 0.976 (self-judged); registry v1 @baseline, v2 @production; alias rollback 18 s / 22 s
- Reviews: python (2 HIGH, 4 MEDIUM) and mle (3 HIGH, 7 MEDIUM) — all fixed or answered in reports/module-3.md "Code review"
- Actions taken:
  - Research agents: Jev via OpenRouter `/api/v1/systemone` (jev-router is a chat router); RAGAS 0.4.3 + langchain-community 0.4.1 pin; MLflow 3.17 `-full` image + `--allowed-hosts`; one-call Jev probe before coding
  - Golden set (56 Q, read from article text), exact metrics, RAGAS wrapper, judge calibration, experiment runner (retrieval-only, from-predictions), MLflow server + tracking + registry, decider ABC + Jev + local cross-encoder, gate sweep, config by alias in the API
  - OpenRouter 402 (no purchased credits; free models 50 req/day) → user chose free route → Gemini API free tier for generation + judge; Jev still works via OpenRouter (~$0.00002/call)
  - Gemini limits learned from 429 bodies: flash-lite 15/min; "3.5-flash" = gemini-3.6-flash 5/min and 20/day → judge moved to flash-lite (same model as generator: self-preference not measurable, calibration is the safeguard); pacer added
  - User labelled 20 answers: all supported → synthetic negatives added so kappa means something
  - Ruling: RAGAS limited to faithfulness + context recall on the free tier — context precision duplicates exact hit@k/MRR (gold labels) and costs one judge call per article — cost if wrong: low
  - Ruling: gate threshold chosen per decider from the offline sweep (Jev 0.5 = middle of the 0.04–0.79 gap), not copied from the article — cost if wrong: low
  - Ruling: API tolerates an unreachable MLflow registry at startup (keeps env settings, logs) — tracking-server outage must not stop answering — cost if wrong: low
  - Ruling: openai held at 3.3.0 by ragas→instructor→jiter<0.15 — only stable OpenAI features used — cost if wrong: low

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
| Phase 2 pytest | `uv run pytest` | all pass, cov >= 80% | 162 passed, 92.12% | pass |
| index build | `dvc repro` on empty Qdrant 1.19.2 | 2,241 points, deterministic name | articles_fb16830c7f, 2,241 points (same name as the build on 1.16) | pass |
| index reuse | `dvc repro` with nothing changed | seconds, no model load | 29 s total, `index_reused` | pass |
| bge-m3 parity | `uv run --with FlagEmbedding python scripts/parity_bge_m3.py` | dense < 1e-4, sparse equal | 1.0e-6 / 0 / 0 mismatches | pass |
| retrieval spot check | 10 concepts × AR/EN + 2 refs | baseline numbers | AR 7/10 @1, 9/10 @5, MRR 0.775; EN 7/10, 10/10, 0.833; refs 2/2 | recorded |
| alias rollback | swap to previous and back | no downtime | 213 ms / 70 ms, 2,241 points served | pass |
| API container | compose core, /live /health /metadata /feedback /ask | 200s, 422 readable | all as expected; runs as uid 1000; healthy in 22 s (warm volume) | pass |
| /ask end to end | 9 questions, prompt v3 | gold cited, refusals | 8/9 (Arabic 147 = retrieval miss), $0.00126 total, p50 2.9 s, TTFT 1.5 s | pass |

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
