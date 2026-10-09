# Task Plan: legal-rag — Arabic Legal Q&A RAG (JEV-RAG), MLOps Practitioner Track B

## Goal

Ship a production-style, fully observable RAG service over the Egyptian Civil Code that answers AR/EN legal questions with article citations, uses Jev for routing/rerank/gate/validation, and covers every course concept — plus a bilingual HTML walkthrough that teaches each concept and each code file.

## Next Step

Phase 0: push branch module-0-bootstrap, open PR #1, wait for green CI, merge. Then start Phase 1 (corpus parser, TDD).

## Current Phase

Phase 0

## Phases

Full detail (tasks, acceptance checks, decisions, risks) lives in `docs/PLAN.md`. Repo: `C:\00-Shobaki\legal-rag` → github.com/ahmedsh711/legal-rag. One branch + PR + tag per phase.

### Phase 0: Bootstrap + concept walkthrough (branch module-0-bootstrap)

- [x] Repo created, corpus + research digests in place, Docker Desktop, gh auth
- [x] uv project (py3.12, src layout), ruff, pre-commit, pytest cov gate 80%, settings.py, .env.example
- [x] docker/compose.yaml skeleton with profiles (core, tracking now; others added in their phases)
- [x] CLAUDE.md, README skeleton, planning files
- [x] docs/walkthrough: index + 4 concept pages (00a foundations, 00b rag-jev, 00c eval-tracking, 00d serving-monitoring), EN/AR/Both toggle, chosen/rejected tables
- [x] minimal CI (lint + tests) added early so PR #1 has a real check
- [ ] PR #1 merged
- **Status:** in_progress

### Phase 1: Corpus pipeline PDF → articles.json + DVC (branch module-1-corpus)

- [ ] TDD ingest/parse.py, normalize.py, validate.py (1,149 articles, repealed 55–80 & 389–417, Art 452 EN empty)
- [ ] dvc init + MinIO remote + dvc.yaml parse→validate; dvc repro idempotent; 20-article spot check in reports/module-1.md
- [ ] walkthrough 01-corpus.html
- **Status:** pending

### Phase 2: Vanilla RAG + FastAPI + Docker (tag v0.1.0)

- [ ] index/build.py (bge-m3 CPU dense+sparse → Qdrant alias + index metadata), retrieval.py hybrid RRF + filters
- [ ] generation.py (OpenAI-compatible, OpenRouter default), pipeline.ask(), prompt with [Art. N] citations + refusal rule
- [ ] FastAPI /ask (+SSE), /health, /metadata, /feedback; lifespan load-once; correlation-id; JSON logs; 422/500 handlers
- [ ] tests ≥80% cov; multi-stage non-root Dockerfile; compose profile core; push image; README 3 commands
- [ ] walkthrough 02-rag-api.html; reports/module-1.md; PR; tag v0.1.0
- **Status:** pending

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

## Notes

- Re-read this file + docs/PLAN.md before each phase; update Next Step on every status change.
- Log every error here; never repeat a failed action unchanged.
