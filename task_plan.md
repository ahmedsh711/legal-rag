# Task Plan: legal-rag — Arabic Legal Q&A RAG (JEV-RAG), MLOps Practitioner Track B

## Goal

Ship a production-style, fully observable RAG service over the Egyptian Civil Code that answers AR/EN legal questions with article citations, uses Jev for routing/rerank/gate/validation, and covers every course concept — plus a bilingual HTML walkthrough that teaches each concept and each code file.

## Next Step

Phase 2: open PR #3 (module-2-rag-api), wait for CI, merge, tag v0.1.0. Then Phase 3: golden set + to_label.csv, RAGAS 0.4 (check docs first), MLflow stack, Decider (Jev via OpenRouter `typesafe/jev-router` — verify the API), reranker ablation vs the Phase 2 baseline numbers in reports/module-2.md. Do not edit DVC stage deps while a stage runs.

## Current Phase

Phase 2

## Phases

Full detail (tasks, acceptance checks, decisions, risks) lives in `docs/PLAN.md`. Repo: `C:\00-Shobaki\legal-rag` → github.com/ahmedsh711/legal-rag. One branch + PR + tag per phase.

### Phase 0: Bootstrap + concept walkthrough (branch module-0-bootstrap)

- [x] Repo created, corpus + research digests in place, Docker Desktop, gh auth
- [x] uv project (py3.12, src layout), ruff, pre-commit, pytest cov gate 80%, settings.py, .env.example
- [x] docker/compose.yaml skeleton with profiles (core, tracking now; others added in their phases)
- [x] CLAUDE.md, README skeleton, planning files
- [x] docs/walkthrough: index + 4 concept pages (00a foundations, 00b rag-jev, 00c eval-tracking, 00d serving-monitoring), EN/AR/Both toggle, chosen/rejected tables
- [x] minimal CI (lint + tests) added early so PR #1 has a real check
- [x] PR #1 merged (CI green: lint 6s, test 7s); branch protection requires lint+test
- **Status:** complete

### Phase 1: Corpus pipeline PDF → articles.json + DVC (branch module-1-corpus)

- [x] TDD ingest/parse.py, normalize.py, validate.py, params.py (1,149 records; repealed 54–80 & 389–417 from PDF notes; 5 documented anomalies)
- [x] dvc init + MinIO remote + dvc.yaml parse→validate; dvc repro idempotent; wipe-and-pull restores identical hashes; 20-article spot check in reports/module-1.md
- [x] python-reviewer + mle-reviewer findings fixed (see reports/module-1.md "Code review")
- [x] walkthrough 01-corpus.html
- [x] PR #2 merged (CI green: lint 8s, test 21s)
- **Status:** complete

### Phase 2: Vanilla RAG + FastAPI + Docker (tag v0.1.0)

- [x] index/build.py (bge-m3 CPU dense+sparse, pinned revision → Qdrant alias + index metadata), retrieval.py hybrid RRF + filters; real index 2,241 points
- [x] generation.py (OpenAI-compatible, OpenRouter default), pipeline.ask(), prompt with [Art. N] citations + refusal rule
- [x] FastAPI /ask (+SSE), /health, /metadata, /feedback; lifespan load-once; correlation-id; JSON logs; 422/500 handlers
- [x] tests ≥80% cov (162 passed, 92.1%); multi-stage non-root Dockerfile; compose profile core; README 3 commands
- [x] bge-m3 parity vs FlagEmbedding (PARITY OK); retrieval spot check (AR MRR 0.775, EN 0.833); alias rollback 213 ms / 70 ms
- [x] real /ask end to end through the container (prompt v1→v3, 8/9 correct, $0.00014/question); image pushed as ahmedshobaki/legal-rag-api:0.1.0
- [x] walkthrough 02-rag-api.html
- [x] reports/module-2.md; reviewers (python, fastapi, security) fixed
- [ ] PR #3 green + merged; tag v0.1.0
- **Status:** in_progress

### Phase 3: Eval, MLflow, Decider + JEV ablation (tag v0.2.0)

- [ ] golden_set.jsonl ≥50 Q (25 AR + 25 EN mirrored, categories) + to_label.csv for 20 human faithfulness labels
- [ ] eval/ragas_run.py (RAGAS 0.4, per-language) → MLflow; eval/judge.py calibration + bias tests
- [ ] MLflow stack (tracking profile); ≥6 runs: chunking, top-k, embedding, prompt; rag_config artifact; registry alias production
- [ ] decider/base.py, jev.py, local.py + tests; eval/ablation.py none/local/jev; AR-vs-EN agreement test
- [ ] walkthrough 03-eval-mlflow-jev.html; reports/module-2.md; PR; tag v0.2.0
- **Status:** pending

### Phase 4: Production serving, CI/CD, load (tag v0.3.0)

- [ ] Decider wired into pipeline; guardrails as metrics (Presidio AR, injection, off-topic, schema) with latency/FPR
- [ ] Redis token bucket 429+Retry-After; vLLM profile llm + LLM_BACKEND switch; awq_eval.py TTFT table; BentoML service
- [ ] ci.yml lint→test→RAGAS smoke gate→build→push SHA tag; branch protection; badge; red/green screenshots
- [ ] Locust 50–100 users p50/p95/p99 + TTFT; bottleneck + one tuning change; inference-pattern table
- [ ] walkthrough 04-serving-ci.html; reports/module-3.md; PR; tag v0.3.0
- **Status:** pending

### Phase 5: Observability, drift, Langfuse (tag v0.4.0)

- [ ] metric contract + prometheus-client instrumentation (stage histograms, decision_total, tokens_total, multiprocess)
- [ ] monitoring profile: Prometheus (relabel, recording rules), node_exporter, cAdvisor, blackbox, Grafana as code 4 rows, Alertmanager ≥6 rules + webhook
- [ ] llm-observability profile: Langfuse v3 traces/spans/scores, prompt labels + rollback, cost/TTFT, online sampling
- [ ] drift_stats.py (KS/chi2/Wasserstein/JS/MMD) + tests; simulate_drift.py; Evidently → Postgres → Grafana → CI gate; storm guard
- [ ] prediction-event schema; Loki+Promtail; walkthrough 05-observability.html; reports/module-5.md; PR; tag v0.4.0
- **Status:** pending

### Phase 6: Automation — Terraform, CT, Airflow, canary, k8s (tag v0.5.0)

- [ ] infra/terraform docker provider (MinIO, Postgres); destroy→apply
- [ ] ct-reindex.yml schedule + dispatch + repository_dispatch, eval gate, Environment approval
- [ ] Airflow reindex_dag + monitoring_dag (branch → repository_dispatch)
- [ ] nginx canary 95/5 + promote.sh + rollback.sh + watcher.py + shadow mirror; strategy table
- [ ] serving/k8s manifests on Docker Desktop k8s; walkthrough 06-automation-release.html; reports/module-4.md; PR; tag v0.5.0
- **Status:** pending

### Phase 7: Optimization write-up, runbook, README, final polish (tag v1.0.0)

- [ ] optimization/benchmark.py journey table (API model vs AWQ-1.5B), semantic cache + hit rate, cost model
- [ ] docs/runbook.md, 10 anti-patterns, docs/architecture.md, README full, closed-loop screenshot chain, maturity self-assessment
- [ ] walkthrough 07 + index; every src file covered EN+AR; glossary; PR; tag v1.0.0
- **Status:** pending

## Key Questions

1. Does Jev handle Arabic queries well enough? → measured in Phase 3 (AR/EN agreement); fallback = translate query or LocalDecider.
2. Is OpenRouter + Jev key available in `.env`? → user provides `OPENROUTER_API_KEY` (and optional `TYPESAFE_API_KEY`) before Phase 2 generation tests.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Repo on Windows `C:\00-Shobaki\legal-rag`, services in Docker Desktop (WSL2 backend) | user choice; keeps this session; no spaces in path |
| Python 3.12 via uv | RAGAS/Evidently/torch lag on 3.13/3.14 |
| Chunk = one article, bge-m3 CPU, Qdrant hybrid + aliases | handbook + Arabic RAG paper; GPU reserved for vLLM |
| Decider ABC: JevDecider + LocalDecider fallback | JEV-RAG pattern + keyless operation + ablation |
| OpenRouter default generation/judge, vLLM Qwen2.5-1.5B-AWQ switchable | Arabic quality vs handbook vLLM/AWQ boxes |
| No Makefile | handbook rule: real commands in README |
| Walkthrough = plain HTML per phase with EN/AR/Both toggle | user choice; no build tool |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Windows `gh` not logged in | 1 | reused WSL gh token via `gh auth login --with-token` |
| `bash` on PATH resolves to WSL bash → ralph-loop hook would fail | 1 | patched plugin hooks.json to Git Bash full path |
| ralph-loop stop hook needs `jq` (missing on Windows) | 1 | installed jq 1.8.1 into ~/.local/bin |
| minio/minio and quay.io MinIO images no longer pullable | 1 | switched to frozen `bitnamilegacy/minio:2025.7.23` (auto-creates buckets); noted in report |
| plain `dvc push/pull` skipped the PDF pointer | 2 | root cause: root .gitignore `data/raw/*` made DVC treat the folder as ignored; removed it, DVC writes per-file .gitignore; wipe-and-pull test passes |
| parser: tanween-ending words broke AR/EN line split | 1 | `_LAST_ARABIC` includes diacritics; regression test added |
| C: drive full (FlagEmbedding pulled the whole bge-m3 repo incl. 2.2 GB ONNX) | 1 | stopped it; deleted only own caches (partial ONNX, duplicate .bin, uv prune, builder image); parity now loads a pinned local snapshot; compose can bind the host HF cache (`HF_CACHE`) |
| transformers 5 silently loaded bge-m3 safetensors from a PR branch | 1 | pinned `embedding_revision` (params + settings), in collection hash + index metadata; API refuses a mismatch |
| `dvc repro` skipped index after code/params edits | 1 | root cause: deps were edited while the stage ran and DVC hashed them at the end; `dvc repro -f -s index`; rule: no dep edits during a run |
| CRLF on Windows → DVC output md5 differs from Linux | 1 | `newline="\n"` in every writer + tests; ruff `line-ending = "lf"`; pre-commit `mixed-line-ending --fix=lf` |
| heredoc/sed escapes turned `\r`/`\n` into real control chars | 2 | edit files with the Edit tool or a scratchpad script, never escapes inside heredocs |
| parity FAILED on sparse weights | 1 | reference bug: FlagEmbedding needs colbert_linear.pt next to sparse_linear.pt or it uses a random head; added to allow_patterns → PARITY OK |
| qdrant-client 1.19 vs server 1.16 warning | 1 | server pinned v1.19.2; volume recreated + index rebuilt (proved always_changed) |
| retrieval 2.2 s per query after binding ports to 127.0.0.1 | 1 | Windows `localhost` tries ::1 first (2,069 vs 17 ms per call) → 127.0.0.1 in all host URLs |
| C: full again (pagefile 16 GB + Docker vhdx growth) | 1 | user freed space (51→62 GB free); Docker engine went read-only → restarted Docker Desktop |
| HF cache bind mount not writable by uid 1000 | 1 | Windows bind mounts are root:root 755 in the container → option removed, named volume only |
| pre-commit mixed-line-ending exe hung (antivirus) | 1 | hook removed; .gitattributes eol=lf + ruff LF + newline="\n" writers cover it |
| port 8000 taken by the user's own studio.py | 1 | API_HOST_PORT in compose (user's .env: 8010) |

## Notes

- Re-read this file + docs/PLAN.md before each phase; update Next Step on every status change.
- Log every error here; never repeat a failed action unchanged.
