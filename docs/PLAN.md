# Plan: Arabic Legal Q&A RAG (JEV-RAG) — MLOps Final Project, Track B

**Mode**: architectural, greenfield. **Complexity**: Large (~100 h of agent work, run phase-by-phase with ralph-loop).
**Repo location (changed by user after approval)**: `C:\00-Shobaki\legal-rag` on Windows (no spaces). This session does the whole build: Python package + tests run with uv/Python 3.12 on Windows; every service (Qdrant, Redis, MLflow, Postgres, MinIO, vLLM, Airflow, Langfuse, Prometheus, Grafana, nginx) runs in Docker Desktop on its WSL2 backend. GitHub repo: `ahmedsh711/legal-rag` (public).

## Context

The user is a junior ML engineer finishing the ITI × MLOps MENA "MLOps Practitioner" course (instructor Aya Nasser Salama). The handbook (References/The-MLOps-Practitioner-Handbook-3-tracks.pdf) defines Track B: a RAG system over the Egyptian Civil Code (bilingual AR/EN PDF in `data/`) that answers legal questions with article citations, graded on a 10-row rubric (packaging, API, Docker, MLflow, DVC, CI/CD, serving, monitoring, peer review, README). The user wants this to be a training project that applies every concept from the 4 sessions, the instructor's 90 LinkedIn posts and the support decks (CI/CD, Docker, FastAPI, Kubernetes), built like a real production system but sized for a junior, and preceded by a bilingual (Egyptian Arabic + English) HTML walkthrough that explains every concept, every design choice (chosen vs rejected and why), and every code file.

"jev+rag" = the JEV-RAG pattern (Gao Dalie, Sept 2026; Thoughtworks): use Jev, TypeSafe AI's cheap non-generative "decision model", for the *deciding* steps of RAG (routing, reranking, answerability gate, claim validation) and keep the LLM only for generation. User also wants the best of the other interpretations folded in (strong eval layer, full dev lifecycle).

### Decisions already made with the user
| Topic | Decision |
|---|---|
| LLM backend | Hybrid. OpenRouter (one key: generation model + RAGAS judge + Jev `typesafe/jev-1.13`) via an OpenAI-compatible client; local vLLM (`Qwen/Qwen2.5-1.5B-Instruct-AWQ`, Docker, GPU) as a switchable second backend for the handbook's vLLM/AWQ/TTFT boxes |
| Scope | Rubric core + every taught concept in its smallest real form. Airflow, Terraform, Kubernetes minimal. No "everything in the handbook at full depth" |
| Environment | WSL2 Ubuntu + Docker Desktop (verified: uv 0.12, git, gh, nvidia-smi RTX 3050 4 GB, 11 GB RAM visible to WSL, no `.wslconfig`, no `claude`/`node` in WSL yet) |
| Accounts | GitHub + Docker Hub exist; create the public repo with `gh` |
| Walkthrough | One HTML page per phase under `docs/walkthrough/`, each section has EN / AR(Egyptian) / Both toggle; plain HTML + shared CSS/JS, no build tool |
| Pace | Agent works autonomously until done (ralph-loop per phase) |

## Research findings that shape the design (done, do not redo)

- **Corpus** (agent inspection): 170 pages, text layer OK, ~1,149 articles, ~300K tokens. Use **pypdf** (PyMuPDF breaks lam-alef, pdfplumber reverses Arabic). Split on English `^Article\s*(\d+)$` lines (Arabic numerals come out digit-reversed, English is authoritative; `Article1022` has no space). Arabic text of an article precedes its English header (Art. 83 flipped). Repealed gaps: 55–80 (noted in PDF), 389–417 (silent). Art. 452 has no English. Mirrored brackets `)١ (` → `(1)`. Carry الكتاب/الباب/الفصل + BOOK/PART/SECTION headings forward as metadata.
- **Arabic RAG** (arXiv 2506.06339, RAGAS-evaluated): bge-m3 (70.99) ≈ multilingual-e5-large (70.31) best embeddings, beating Arabic-specific models; bge-reranker-v2-m3 lifts faithfulness most on complex corpora (+15 pts on ARCD); sentence/unit-aware chunking beats fixed-size. Handbook mandates **chunk = one article** (natural citation unit, ~180–270 tokens).
- **Jev** (docs.typesafe.ai, OpenRouter guide): `POST /v1/systemone`, `pip install typesafe-sdk`, `TypeSafeClient(api_key).evaluate(state=..., model="jev-1.13.0", questions={id: {type: noul|choice|score, instructions, criteria}})`; response `answers[id]` with `noul` prob / `choice`+`probabilities`+`confidence` / `score`+`probabilities`+`confidence`; 32k-token state budget, ~70–500 ms, $0.042/M input tokens, output free; 429/529 → backoff. English primary; other languages "handled but not equally well" → always send English article text in state, test Arabic queries explicitly. Also reachable through OpenRouter (`typesafe/jev-1.13`, `POST /api/v1/systemone`).
- **vLLM on RTX 3050 4 GB** (DamnKuldeep repo): `vllm/vllm-openai` pinned tag, AWQ+Marlin, `--gpu-memory-utilization 0.78` (WDDM reserves VRAM), weights 1.1 GiB, ~180 tok/s at 20 users, set NVIDIA Control Panel "Prefer maximum performance" (default power policy = 11× throughput loss). We use `--max-model-len 4096 --max-num-seqs 8`.
- **Library churn to respect**: RAGAS 0.4.x (`ragas.metrics.collections`), Evidently ≥0.7 (new `Report`/`Tests` API), Langfuse v3 self-host (web+worker+Postgres+ClickHouse+Redis+MinIO, ~4 GB RAM), MLflow prompt registry exists (≥2.12). Always consult context7 before writing against these.
- **Course concept checklist to cover** (from transcripts/posts/decks): maturity model, src layout + pyproject + uv lock, OOP abstraction (ABC with swappable impls), pydantic validation + 422, load-once lifespan, async + run_in_threadpool, structured JSON logs + correlation id, pytest fixtures/mocks/parametrize + 80% gate, multi-stage non-root Docker + .dockerignore + compose, MLflow runs/registry/aliases ("load by alias, never by path"), DVC add/remote/dvc.yaml/repro + lineage (git SHA → DVC hash → MLflow run), ruff + pre-commit, GitHub Actions (lint→test→gate→build→push SHA tag, secrets, environments, schedule/dispatch/repository_dispatch), Terraform (docker provider), Airflow DAG (branch, sensor, XCom refs, retries), inference patterns decision table, serving categories + BentoML, Locust p50/p95/p99 + TTFT, canary/blue-green/A-B/shadow with rollback, Prometheus pull model + metric contract + cardinality budget + multiprocess, PromQL, Grafana as code + annotations, Alertmanager symptom-vs-cause + for: + routing, Evidently + Postgres drift store + CI gate + retrain branch + storm guard, three pillars + prediction-event schema, Langfuse traces/spans/generations/scores + prompt labels + cost/TTFT, RAGAS 4 metrics + calibrated judge + bias tests + ablation + CI gate + online sampling, guardrails as metrics (PII/injection/off-topic/groundedness) with measured cost and FPR, Redis token bucket + honest 429, vLLM internals (PagedAttention, continuous batching, KV cache) + AWQ + VRAM table, drift taxonomy incl. embedding/corpus/retrieval/prompt/provider drift, runbook + anti-patterns, honesty clause.

## Target architecture (what exists at the end)

```
user ──HTTP──▶ nginx (canary 95/5, shadow mirror) ──▶ FastAPI /ask (SSE) ──┐
                                                                            │ pipeline.ask()
   route(Jev choice) → retrieve(Qdrant hybrid bge-m3, filters) → rerank(Jev score ×N parallel)
   → gate(Jev noul ≥0.75 else widen→refuse) → generate(OpenRouter | vLLM) → validate claims(Jev noul)
   → guardrails (metrics) → response {answer, sources:[Art. N], decisions[], request_id}
                                                                            │
   Langfuse trace (spans per stage, generation cost, scores) ◀──────────────┤
   Prometheus /metrics (stage histograms, decision_total, tokens_total) ◀───┘
   MLflow (chunking/embedding/decider experiments, registered rag_config) · DVC (pdf→articles.json→index)
   Airflow (reindex_dag, monitoring_dag → repository_dispatch CT) · GitHub Actions (CI + CT with approval)
   Prometheus+Grafana+Alertmanager · Evidently drift → Postgres → Grafana · Redis (rate limit, semantic cache)
```

`Decider` is an ABC with `JevDecider` (default when `TYPESAFE_API_KEY`/OpenRouter set) and `LocalDecider` (bge-reranker-v2-m3 cross-encoder + LLM yes/no) so the project runs keyless and the two can be A/B-compared with RAGAS + Langfuse scores. This is the course's "ModelBase ABC" idea applied to RAG.

## Repo layout (WSL: `~/projects/legal-rag`)

```
legal-rag/
  README.md  pyproject.toml  uv.lock  .python-version(3.12)  .pre-commit-config.yaml  .env.example
  .gitattributes(eol=lf)  .gitignore  .dockerignore  dvc.yaml  params.yaml
  task_plan.md  findings.md  progress.md            # planning-with-files
  data/raw/egyptian_civil_code.pdf(DVC)  data/processed/articles.json(DVC)  data/golden/golden_set.jsonl
  src/legalrag/
    settings.py  logging_conf.py
    ingest/{parse.py,normalize.py,validate.py}
    index/{build.py,store.py}            retrieval.py
    decider/{base.py,jev.py,local.py}    generation.py  pipeline.py  guardrails.py  ratelimit.py  cache.py
    api/{main.py,schemas.py,middleware.py}
    observability/{metrics.py,tracing.py}
    eval/{golden.py,ragas_run.py,judge.py,ablation.py}
    monitoring/{drift_stats.py,drift_job.py,simulate_drift.py}
  tests/{conftest.py,unit/,integration/,eval/}
  docker/{compose.yaml,Dockerfile,nginx/,vllm/}
  pipelines/{dvc.yaml→root, airflow/dags/{reindex_dag.py,monitoring_dag.py}}
  .github/workflows/{ci.yml,ct-reindex.yml}
  infra/terraform/{main.tf,variables.tf,outputs.tf,versions.tf}
  serving/{bento/{service.py,bentofile.yaml}, canary/{rollback.sh,promote.sh,watcher.py}, k8s/*.yaml}
  loadtest/locustfile.py
  optimization/{awq_eval.py,vram_table.md,benchmark.py}
  monitoring/{prometheus/prometheus.yml,rules/{alerts.yml,recording.yml},grafana/provisioning/,alertmanager/,blackbox/}
  reports/module-1.md … module-5.md  (+ screenshots/)
  docs/{runbook.md,architecture.md,walkthrough/{assets/{style.css,toggle.js},00-concepts.html,01-…html}}
```

No Makefile (handbook rule: real commands in README).

## Phases (each = branch `module-N-*` → PR → tag; each ends demoable)

Rubric rows: 01 packaging · 02 API · 03 Docker · 04 MLflow · 05 DVC · 06 CI/CD · 07 serving · 08 monitoring · 09 peer review · 10 README.

### Phase 0 — Bootstrap + research-backed walkthrough (no product code)
1. **From this Windows session after approval**: `gh repo create legal-rag --public` (under the user's GitHub); via `wsl.exe` create `~/projects/legal-rag`, write `C:\Users\…\.wslconfig` (`memory=16GB`, `swap=8GB`), copy this plan as `task_plan.md` seed + `KICKOFF.md` (the prompt to paste into the WSL session). Save memory notes (user profile, project decisions).
2. **In WSL session**: install Node + Claude Code; install plugins: `superpowers`, `ralph-loop`, `context7`, `planning-with-files`, `ponytail` (optional) + marketplace `mlflow`, `qdrant-skills`, `astronomer-data-agents` (Airflow), `redis-development`, `terraform`, `huggingface-skills`; patch ralph-loop hook if needed (README's Windows note is irrelevant in WSL).
3. `uv init --lib --python 3.12`, src layout, ruff, pre-commit (ruff, ruff-format, end-of-file-fixer, check-yaml, detect-secrets), pytest + `--cov-fail-under=80`, `settings.py` (pydantic-settings) + `.env.example`, `docker/compose.yaml` skeleton with profiles, CLAUDE.md (project rules: TDD, no Makefile, no print, load-once, etc.), README skeleton.
4. **Walkthrough concept pages** (written before code, from the digests): `00-concepts.html` covering MLOps maturity & why; RAG anatomy; chunking (article vs fixed vs sentence — chosen/rejected); embeddings (bge-m3 vs e5 vs Arabic-specific); hybrid search + RRF; reranking (cross-encoder vs Jev); JEV-RAG (what Jev is, choice/score/noul, where it replaces the LLM, tiers 0.9/0.6, Arabic caveat, fallback); generation backends (OpenRouter vs vLLM, AWQ, PagedAttention, KV cache, TTFT); evaluation (RAGAS 4 metrics and what each blames, judge calibration, bias tests, golden set); tracking & versioning (MLflow, DVC, lineage); CI/CT; serving patterns & categories; load testing; release strategies; monitoring (Prometheus contract, cardinality, Grafana as code, alerts symptom/cause); drift taxonomy incl. RAG drifts; Langfuse; guardrails & rate limits; runbooks. Each concept: simple EN + Egyptian AR, instructor's phrasing/analogies where available, "what we chose / what we rejected / why".
   Acceptance: `uv run pytest` green on a smoke test, `pre-commit run -a` clean, `docker compose config` valid, walkthrough opens in a browser with working toggle, PR #1 merged.

### Phase 1 — Corpus: PDF → articles.json (rubric 05 start) — tag v0.1.0 together with Phase 2
- TDD `ingest/parse.py` (pypdf, join pages, split on English Article lines, heading stack → book/chapter/section/topic, pair preceding Arabic, bracket fix, repealed flags for 55–80 & 389–417, `source_page`, `citation`), `ingest/normalize.py` (alef/yaa/taa-marbuta/tatweel/diacritics, Arabic-Indic digits → int), `ingest/validate.py` (pydantic record; contiguity except known gaps; non-empty text_ar; max length; repealed flagged) run as a DVC stage that fails on bad output.
- `dvc init`, MinIO remote (compose profile `tracking`), `dvc add` PDF, `dvc.yaml` stages `parse → validate`; `dvc repro` idempotent; spot-check 20 random articles and record findings in `reports/module-1.md`.
- Walkthrough `01-corpus.html`: extraction pitfalls (digit reversal, lam-alef, columns), why chunk-by-article, DVC content-addressing, lineage.

### Phase 2 — Vanilla RAG + FastAPI + Docker (rubric 01, 02, 03) — tag v0.1.0
- `index/build.py`: bge-m3 on **CPU** (GPU reserved for vLLM; one-off ~10 min), dense + sparse vectors, two points per article (AR text, EN text) sharing payload; Qdrant collection `articles_v<hash>` + alias `articles`; index metadata record `{embedding_model, normalization_version, built_at, git_sha, dvc_hash}`; API refuses to start on mismatch (instructor's incident).
- `retrieval.py` hybrid RRF + payload filters (`book`, `is_repealed=false` default) → `Chunk` objects. `generation.py` OpenAI-compatible client (`LLM_BASE_URL/LLM_MODEL/LLM_API_KEY`), timeout, prompt with inline `[Art. N]` citations and "refuse if not in context" rule; `pipeline.py` `ask()` retrieve → generate.
- FastAPI: `/ask` (JSON; `stream=true` SSE), `/health` (`{status, documents_indexed, index_version}`), `/metadata`, `/feedback`; Pydantic `Field(min_length=3, max_length=2000)` → 422; lifespan load-once (Qdrant client, embedder, config); `run_in_threadpool` for CPU embed; correlation-id middleware; JSON logs (structlog) with `request_id, stage, duration_ms, model_version`; exception handlers (422 clean, 500 no traceback leak).
- Tests: conftest fixtures (sample articles, fake Qdrant, mocked LLM), `test_parse`, `test_normalize`, `test_retrieval`, `test_api` (200/422/health/schema contract), parametrize edge cases; coverage ≥80%.
- Multi-stage `Dockerfile` (uv, non-root, HEALTHCHECK), `.dockerignore`, compose profile `core` (qdrant, redis, api); push `ahmedshobaki/legal-rag-api:0.1.0` + `latest`; README 3 commands.
- Walkthrough `02-rag-api.html`: every file + function, Pydantic/lifespan/async/logging/Docker choices.

### Phase 3 — Evaluation, MLflow, Decider + JEV ablation (rubric 04) — tag v0.2.0
- `data/golden/golden_set.jsonl`: ≥50 questions (25 AR + 25 mirrored EN; categories in_scope / repealed-article / off-topic / injection), gold article ids + reference answers; 20 hand-labelled faithfulness rows for judge calibration (user labels them; agent prepares the sheet).
- `eval/ragas_run.py` (RAGAS 0.4 collections API: faithfulness, answer relevancy, context precision, context recall; judge via OpenRouter; per-language reporting) → MLflow run; `eval/judge.py` own rubric judge + calibration (agreement vs 20 labels) + position/verbosity/self-preference bias tests.
- MLflow stack (profile `tracking`: postgres + minio + mlflow) with experiments: chunking (article vs paragraph-split), top-k {3,5,8}, embedding {bge-m3, multilingual-e5-large}, prompt versions — every run logs params + metrics + `rag_config.json` artifact + tags `git_sha, dvc_hash`. ≥6 runs; screenshot.
- `decider/base.py` (`Decision` dataclass, `Decider` ABC: `route`, `rerank`, `can_answer`, `validate`), `decider/jev.py` (typesafe-sdk/OpenRouter, httpx timeout 2 s, parallel `score` per chunk, `noul` gate/validate, English article text in state), `decider/local.py` (book-centroid routing, bge-reranker-v2-m3 cross-encoder on CPU, LLM JSON yes/no); thresholds in settings (`TIER_ACT=0.9`, `TIER_SECOND_OPINION=0.6`, `GATE_THRESHOLD=0.75`, `VALIDATE_THRESHOLD=0.75`, `RERANK_KEEP_TOP=5`); fallback Jev→Local on `DeciderUnavailable`.
- `eval/ablation.py`: none / Local / Jev on golden set → MLflow; AR-vs-EN agreement test for Jev (if <0.8 enable `JEV_TRANSLATE_QUERY` with cached one-line translation). Register best config in MLflow registry (`legal-rag-config`, alias `production`); API loads config by alias at startup.
- Walkthrough `03-eval-mlflow-jev.html`.

### Phase 4 — Production serving, CI/CD, load (rubric 06, 07) — tag v0.3.0
- Wire Decider into `pipeline.py` (route → retrieve top-12 → rerank → gate → generate → validate → `decisions[]` in response); guardrails as instrumented metrics (Presidio with Arabic recognizers, injection: regex + Jev `noul` incl. doc-embedded check at index time, off-topic, schema, groundedness via validate); measure added p95 and FPR on clean Arabic traffic; adversarial injection set with per-category detection rate.
- Redis token bucket (Lua) per API key → `429 + Retry-After`; upstream 429 backoff with jitter; both as metrics.
- vLLM (profile `llm`, `docker/vllm/`), `LLM_BACKEND=vllm` switch; `optimization/awq_eval.py` TTFT / tokens-per-sec at 1, 5, 20 concurrency; VRAM table; RAGAS API-model vs AWQ-1.5B (document the quality gap honestly).
- BentoML `serving/bento/service.py` wrapping `pipeline.ask` async; `bentoml build && containerize`.
- `.github/workflows/ci.yml`: lint → test (cov ≥80) → RAGAS smoke gate (10 Q, faithfulness ≥0.75, secrets from GitHub Secrets) → buildx push `sha-<short>` + `latest` (docker/login-action); branch protection; CI badge; red-then-green screenshots.
- `loadtest/locustfile.py` realistic question mix (80 % ask AR/EN, 15 % stream, 5 % metadata; `between(1,3)`), 50–100 users, p50/p95/p99 + TTFT; bottleneck diagnosis; one tuning change before/after; inference-pattern decision table in report.
- Walkthrough `04-serving-ci.html`.

### Phase 5 — Observability, drift, LLM observability (rubric 08) — tag v0.4.0
- Metric contract first (`reports/module-5.md`): `rag_requests_total{backend,status}`, `rag_stage_seconds{stage}` histogram (route/retrieve/rerank/gate/generate/validate), `decision_total{stage,decider,outcome}`, `llm_tokens_total{backend,type}`, `rag_inflight`, `rag_info{index_version,config_version}`; cardinality budget; buckets around SLA; multiprocess mode proven.
- Profile `monitoring`: Prometheus (hand-written yml with relabel), node_exporter, cAdvisor, blackbox, vLLM target; recording rules; ≥5 PromQL in report. Grafana provisioned from repo: 4 rows (service / resources / RAG behaviour / drift+quality), `$backend` variable, deploy annotations. Alertmanager ≥6 rules with `for:` + symptom/cause labels + first action: api down, p95 > SLA, 5xx > 1 %, faithfulness < 0.80, daily token budget, drift score, VRAM/KV-cache > 90 %, SLO burn-rate; Discord/Slack webhook; trigger 3.
- Profile `llm-observability`: Langfuse v3 (shares postgres/redis/minio); `@observe` spans per stage with `request_id, prompt_version, top_k, index_version, decider`; prompts moved to Langfuse with `production`/`canary` labels, rollback as label move (timed); custom vLLM pricing; cost per trace; TTFT; RAGAS scores pushed to traces; online sampling (score 5 % of traffic nightly).
- `monitoring/drift_stats.py` KS/chi2/Wasserstein/JS-KL/MMD+domain-classifier with unit tests (no firing on control); `simulate_drift.py` (query-intent shift: contract-law → family-law questions; embedding-model swap); Evidently 0.7 report + tests on query embeddings/text descriptors vs frozen golden reference → `drift_metrics` Postgres table → Grafana SQL row → CI gate; storm guard (cooldown, min samples, daily cap).
- Prediction-event JSON schema (no raw PII) + metric→log drill-down demo; Loki+Promtail (lighter than ELK, justified).
- Walkthrough `05-observability.html`.

### Phase 6 — Automation: Terraform, CT, Airflow, canary, k8s (rubric 06 depth) — tag v0.5.0
- Terraform docker provider (`infra/terraform`, 4 files) for MinIO + Postgres + network/volumes; `destroy → apply` round-trip; tfstate gitignored + paragraph on remote state.
- `ct-reindex.yml`: `schedule` + `workflow_dispatch(data_version)` + `repository_dispatch(drift-detected)`; `dvc pull → repro → eval gate (faithfulness ≥ production − margin) → promote config alias to staging → GitHub Environment approval → production`; never auto-promote to production.
- Airflow (profile `airflow`, LocalExecutor, shares postgres): `reindex_dag` (FileSensor → parse → validate → index → eval → BranchPythonOperator → swap alias / skip, XCom passes run_id only, retries, idempotent) and `monitoring_dag` (daily drift → branch → `repository_dispatch`). Division-of-labour paragraph (Airflow vs Actions).
- nginx canary 95/5 with version header, `promote.sh`, `rollback.sh` (timed), `watcher.py` (Prometheus p95/5xx/faithfulness → rollback), shadow `mirror`; strategy table (blue/green, canary, A/B, shadow).
- `serving/k8s/`: Deployment (probes, requests/limits), Service, canary Ingress annotations; apply on Docker Desktop Kubernetes; `kubectl rollout undo` demo.
- Walkthrough `06-automation-release.html`.

### Phase 7 — Optimization write-up, runbook, README, final polish — tag v1.0.0
- `optimization/benchmark.py` harness (warmup, ≥100 iters, fixed set) → journey table: OpenRouter model vs AWQ-1.5B (quality × TTFT × tok/s × VRAM); semantic cache (Redis, cosine ≥0.95) with hit-rate metric and cost delta; cost model (requests/month → $).
- `docs/runbook.md` (one section per alert, first command, rollback, escalate), 10 anti-patterns, `docs/architecture.md` (mermaid diagram spanning all modules), README full overview + Grafana screenshot + session changelog; closed-loop screenshot chain at top of `reports/module-5.md`; maturity self-assessment.
- Final walkthrough pass: `07-optimization-runbook.html` + index page linking all; every file/function covered; glossary EN↔AR.

## Decider / JEV layer (concrete)
- `Decision(stage, outcome, score, confidence, decider, latency_ms, raw)`; `Decider.route|rerank|can_answer|validate` async.
- Jev state = English: question (as typed) + `"[Art. N] (book / chapter)\n{text_en or text_ar}"` per chunk (≤12 chunks ≪ 32k). `route`: `choice` over books + `off_topic` + `needs_clarification`; `rerank`: `score` levels `["irrelevant","tangential","relevant","directly answers"]` one question per chunk in one request; `gate`: `noul` "fully answerable from these articles alone?"; `validate`: sentence-split answer, one `noul` per sentence → supported ratio (cheap faithfulness proxy, pushed to Langfuse as a score).
- Tiers: >0.9 act; 0.6–0.9 second opinion from `LocalDecider`; <0.6 widen once (top_n×2, drop filter) then refuse with "لا أجد نصًا في القانون المدني يجيب على هذا السؤال" + closest articles.
- Logging: Langfuse span `decider.<stage>` + scores; Prometheus `decision_total{stage,decider,outcome}`, `decision_latency_seconds{stage,decider}`, `decision_fallback_total`.
- `# ponytail:` per-request fallback, no circuit breaker until Jev flaps.

## Compose profiles & resource rules
| Profile | Services | ~RAM |
|---|---|---|
| core | qdrant, redis, api | 1.5 GB |
| tracking | postgres, minio, mlflow | 1 GB |
| llm | vllm (GPU 0.78) | 3 GB + 3.1 GB VRAM |
| monitoring | prometheus, grafana, alertmanager, node_exporter, cadvisor, blackbox, loki, promtail | 2 GB |
| llm-observability | langfuse-web, langfuse-worker, clickhouse | 2.5 GB |
| airflow | webserver, scheduler | 2 GB |
| serving | nginx, api-canary, api-shadow, bento | 1.5 GB |
`.wslconfig memory=16GB`. Runbook rule: never `airflow + llm-observability + llm` together except for the final demo. Embedding + reranker on CPU; GPU belongs to vLLM.

## Tooling for execution (WSL session)
- Skills/plugins: superpowers (brainstorming done; use `writing-plans` → `executing-plans`/`subagent-driven-development`, `test-driven-development`, `verification-before-completion`), planning-with-files (`task_plan.md`, `findings.md`, `progress.md`), ralph-loop per phase, context7 (`docs-lookup` agent) before any RAGAS/Evidently/Langfuse/vLLM/Qdrant API use, ponytail (keep minimal), artifact-design + dataviz skills for the walkthrough HTML, reviewers: `mle-reviewer`, `python-reviewer`, `fastapi-reviewer`, `security-reviewer` at each phase end, `code-reviewer` before each PR.
- Ralph-loop prompt template per phase: "Implement Phase N of legal-rag per task_plan.md §Phase N. TDD, context7 before new library APIs, small commits on branch module-N-…, tick progress.md after each task and paste the acceptance command output, run reviewers, update docs/walkthrough/0N-*.html (EN+AR) and reports/module-N.md with screenshots, open PR, tag. Do not add services/abstractions/deps not in the phase table. Output `<promise>PHASE N DONE</promise>` only when every acceptance check in the phase table is green." with `--max-iterations 30`.
- Secrets: `.env` (gitignored) with `OPENROUTER_API_KEY`, optional `TYPESAFE_API_KEY`, `DOCKERHUB_*` only in GitHub Secrets; `detect-secrets` pre-commit.

## Risks
| Risk | Mitigation |
|---|---|
| 4 GB VRAM | AWQ 1.5B only, `--max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.78`, API backend default, VRAM table documents limits |
| RAGAS judge cost / Arabic weakness | 50-Q set, 10-Q CI smoke, cheap judge via OpenRouter, per-language scores, calibration vs 20 human labels |
| Jev Arabic weakness | English article text in state; AR/EN agreement test gates query translation; Local fallback always works |
| Library API churn (RAGAS 0.4, Evidently 0.7, Langfuse v3) | pin in `uv.lock`; context7 before use; `drift_stats.py` is pure scipy |
| Laptop RAM | profiles + `.wslconfig` + runbook rule; cAdvisor dashboard proves usage |
| Python 3.14 in WSL | `uv python install 3.12`, `.python-version` |
| Scope creep | each phase demoable; optional items (semantic cache, k8s) last; honesty clause: report negative results |
| Hand-labelling needs the user | agent prepares `data/golden/to_label.csv`; phase 3 judge calibration waits on it, everything else proceeds |

## Verification (end-to-end)
1. Fresh clone in WSL: `uv sync && docker compose --profile core --profile tracking up -d && dvc pull && dvc repro && uv run python -m legalrag.index.build && curl -X POST localhost:8000/ask -d '{"question":"ما هي مدة التقادم في الالتزامات؟"}'` returns an answer with `[Art. N]` citations and `request_id`.
2. `uv run pytest --cov` ≥80 %; `pre-commit run -a` clean; CI green on PR; image pullable from Docker Hub.
3. MLflow UI shows ≥6 runs incl. decider ablation; registry alias `production`; swapping alias changes `/metadata` without rebuild.
4. `dvc repro` reproduces `articles.json` hash; `git checkout` older commit + `dvc checkout` reverts data.
5. Locust report (p50/p95/p99, TTFT) in `reports/`; vLLM backend answers via `LLM_BACKEND=vllm`.
6. Prometheus targets all UP; Grafana dashboard survives `docker compose down -v`; 3 alerts fired to webhook; Langfuse trace waterfall with scores; drift simulated → Postgres row → alert → Airflow branch → CT workflow → approval gate (screenshot chain).
7. `rollback.sh` timed; watcher auto-rollback screenshot; `kubectl get pods` Ready.
8. Walkthrough: every `src/` file appears in a code-tour section with EN + AR; toggle works; peer reviewer can follow README cold.

## Immediate next steps after approval (this Windows session)
1. `gh repo create <user>/legal-rag --public` (name confirm: `legal-rag`).
2. Write `.wslconfig` (16 GB) and, via `wsl.exe`, `mkdir -p ~/projects/legal-rag`, drop `task_plan.md` (this plan, adapted), `KICKOFF.md` (exact commands: install Node/Claude Code in WSL, plugin installs, first ralph-loop command), and `.env.example`.
3. Save memory notes (user: junior MLOps student, WSL user `shobaki`; project decisions above).
4. Hand off: user opens Claude Code from WSL in `~/projects/legal-rag` and runs the Phase 0 ralph-loop.
