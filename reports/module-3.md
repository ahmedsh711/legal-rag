# Module 3 report: evaluation, experiment tracking, and the decision layer (JEV-RAG)

**Goal:** turn "it seems to work" into numbers: a golden set, free exact metrics plus an LLM judge, every experiment tracked in MLflow with its data and code versions, and a measured answer to "does a decision model (Jev) or a local reranker make the RAG better, and at what cost?"

## What was built

| Piece | File | Rubric row |
|---|---|---|
| Golden set: 28 AR/EN mirrored pairs (56 questions), 5 categories, checked against the corpus | `data/golden/golden_set.jsonl`, `eval/golden.py` | 04 |
| Exact metrics (hit@k, MRR, citations, refusals, language, latency, cost) + offline gate sweep | `eval/metrics.py` | 04 |
| RAGAS 0.4 judge (faithfulness, answer relevancy, context precision/recall), per-item error tolerance | `eval/ragas_run.py` | 04 |
| Judge calibration vs human labels (agreement, Cohen's kappa) + self-preference and verbosity probes | `eval/judge.py`, `data/golden/to_label.csv` | 04 |
| Experiment runner: one command per experiment, retrieval-only mode, MLflow logging | `eval/run.py` | 04 |
| MLflow 3.17 server (Postgres + MinIO), runs with lineage tags, config registry with alias | `docker/compose.yaml`, `eval/track.py` | 04, 05 |
| Decider ABC + Jev (OpenRouter `/systemone`) + local cross-encoder; rerank + answerability gate in the pipeline | `decider/`, `pipeline.py` | 07 |

## Retrieval ablation (retrieval only, no LLM: 48 answerable questions)

| Run | hit@1 | hit@5 | MRR | MRR AR | MRR EN | latency p50 | decider cost |
|---|---|---|---|---|---|---|---|
| hybrid (dense + sparse, RRF) | 0.812 | 0.979 | 0.892 | 0.958 | 0.826 | 0.5 s | 0 |
| **dense only** | **0.938** | 0.979 | **0.955** | 0.951 | **0.958** | 0.7 s | 0 |
| sparse only | 0.604 | 0.833 | 0.692 | 0.810 | 0.573 | 0.7 s | 0 |
| hybrid + local cross-encoder | 0.896 | 0.979 | 0.924 | 0.972 | 0.876 | **35.9 s** | 0 |
| hybrid + Jev | 0.979 | 0.979 | 0.979 | 1.000 | 0.958 | 0.6 s | $0.0055 |
| **dense + Jev** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.6 s | $0.0055 |

Jev (one OpenRouter `/systemone` request per question: a 0–3 `score` per article + one `noul` gate, articles in English) puts a gold article first for every answerable question, in both languages, for about $0.0001 per question and ~0.5 s. The local cross-encoder is free per call but 60× slower on CPU and ranks worse.

**Arabic vs English (Jev reads English article text; does an Arabic question get the same decision?)** On the 28 mirrored pairs: same top article for all 25 answerable pairs (the 3 differences are off-topic/injection questions, where no article is right), the same gate decision for 28/28 pairs, mean gate-score difference 0.028 (max 0.11). Translating Arabic questions before calling Jev is not needed.

Reading it:
- The sparse (keyword) signal is weak on its own, worst in English (MRR 0.57). Reciprocal-rank fusion gives it the same weight as the dense signal, so on "What is X?" definitions it pulls a neighbouring article above the defining one (936 above 935 for pre-emption, 803 above 802 for ownership, 1033 and 1035 above 1030 for the official mortgage, 121 above 120 for mistake). Dense alone is best overall; for Arabic, hybrid and dense are equal.
- The local cross-encoder improves hybrid ranking (0.892 → 0.924) but does not beat dense alone, and on this laptop's CPU it costs ~9 s of compute per question (36 s p50 with 4 questions queued on one model). Not viable for serving without a GPU.
- 48 answerable questions is a small sample: 0.955 vs 0.892 is 8 questions changing rank. Directionally clear, not a precise estimate.

### The gate, replayed offline at every threshold

| Threshold | Local cross-encoder: false / correct refusals | Jev (dense): false / correct refusals |
|---|---|---|
| 0.10 | **0 % / 100 %** | **0 % / 100 %** |
| 0.30 – 0.50 | 6.2 % / 100 % | **0 % / 100 %** |
| 0.755 (the JEV-RAG article's value) | 10.4 % / 100 % | **0 % / 100 %** |
| 0.80 | 12.5 % / 100 % | 2.1 % / 100 % |
| 0.90 | 37.5 % / 100 % | 8.3 % / 100 % |

- Local cross-encoder: off-topic/injection score ≤ 0.011, the lowest answerable question 0.245: a narrow window (0.05–0.20) that works.
- Jev: off-topic/injection score ≤ 0.04, every answerable question ≥ 0.79: a wide margin. **Chosen gate: 0.5**, the middle of the gap (most room for error on both sides).
- Repealed-article questions name their article, so the gate never applies to them (they score lowest with both deciders).
- **A threshold belongs to one model**: the article's 0.755 is right for Jev and would wrongly refuse one answerable question in ten with the local model.

## End to end (generation + RAGAS judge), Gemini free tier

Generator `gemini-3.1-flash-lite`, judge `gemini-3.1-flash-lite` (RAGAS faithfulness + context recall), 56 questions, paced under 12 requests/minute (zero 429s). Both runs logged to MLflow with lineage tags.

| Metric | baseline (hybrid, no decider) | **dense + Jev, gate 0.5** |
|---|---|---|
| hit@1 / MRR | 0.792 / 0.878 | **1.000 / 1.000** |
| citation recall / precision | 0.979 / 0.804 | **1.000** / 0.783 |
| RAGAS context recall | 0.979 | **1.000** |
| RAGAS faithfulness (AR / EN) | 0.976 (0.986 / 0.965) | 0.969 (0.975 / 0.964) |
| false refusals / correct refusals | 0 % / 100 % | 0 % / 100 % |
| invalid citations, answer language | 0 %, 100 % | 0 %, 100 % |
| decider latency p50 | — | 0.43 s (retrieval 0.24 s) |
| decider cost (56 questions) | $0 | $0.006 |
| LLM tokens per question (prompt / completion) | 1,265 / 64 | 1,087 / 66 |

Reading it:
- **Jev fixes retrieval, not writing.** Every question now gets its defining article first (MRR 1.0, recall 1.0); the answers cite the gold article every time. The prompt is also ~14 % shorter because the five articles shown are the relevant ones.
- **Faithfulness is a tie within judge noise.** Of the four answers the judge scored below 0.9 in the Jev run, two are near-verbatim copies of the gold article (Art. 418 scored 0.75, Art. 935 scored 0.5). A cheap judge marks down correct answers; this is exactly why it is calibrated against human labels (below).
- **Latency is confounded by the free tier.** The whole baseline request had p50 2.0 s; a few hours later, generation *alone* had p50 5.3 s (p95 25 s) in the Jev run, for the same model and similar prompt sizes, while Jev itself added 0.43 s. Comparing latency across runs on a shared free tier says more about the provider's load than about our pipeline; Phase 4's load test measures it properly.
- **Citation precision dipped slightly** (0.804 → 0.783): with the defining article first, the model more often also cites the next article in the same section. Not a correctness problem (all cited articles were shown; invalid citations stayed 0).

## Model registry: the config is served by alias

| Version | Alias | Config | From MLflow run |
|---|---|---|---|
| 1 | `baseline` | no decider, hybrid, gate 0.75 | `e2e-hybrid` |
| 2 | **`production`** | **Jev, dense, gate 0.5** | `e2e-dense-jev` |

- Registered with `python -m legalrag.eval.track register --run-id … --alias …`: the version is exactly the evaluated run's `rag_config.json`.
- The API (`CONFIG_SOURCE=mlflow`) reads `models:/legal-rag-config@production` at startup via two REST calls; `/metadata` shows `"config_source": "legal-rag-config@production (v2)"`, `decider_backend: jev`, `retrieval_mode: dense`, `gate_threshold: 0.5`.
- **Rollback demo:** moving `production` to v1 and restarting the API took 18 s (it then served v1: no decider, hybrid); rolling forward to v2 took 22 s. No rebuild, no new image; most of the time is reloading bge-m3.
- MLflow 3.17 gotcha: `mlflow.artifacts.download_artifacts` hands the client a presigned URL for `http://minio:9000`, which does not resolve outside Docker; the client retried with back-off for 15 minutes. Config reads now stream through the tracking server (`/get-artifact`), the same call the API uses.

{{JUDGE}}

## Problems we hit

1. **`typesafe/jev-router` is not the decision model.** It is a chat-model router. Jev decisions use the TypeSafe request body at OpenRouter's `/api/v1/systemone`. A one-call probe confirmed the format before any code.
2. **Dependency chain.** RAGAS 0.4.3 does not import with the newest langchain-community (pinned 0.4.1), and RAGAS → instructor → `jiter<0.15` holds `openai` at 3.3.0 for the whole lock (the API image too). Our code only uses long-stable OpenAI features; to revisit when instructor relaxes the pin.
3. **MLflow global state in tests.** The active experiment id leaked between tests and artifacts were written to `./mlruns` in the repo; each test now gets its own directory and experiment.
4. **The gate sweep counted the run's own gate.** A first sweep reported a flat 0.9 at every threshold because refusals made by the run's 0.75 gate were kept. Predictions now record who refused (`gated` vs the LLM) and the sweep replays only the gate.
5. **Out of credits.** OpenRouter answered `402` during the first RAGAS run: the account had never bought credits, and free models are capped at 50 requests/day (one golden run = 56; with RAGAS ≈ 600). Predictions were already saved, so nothing was lost. Generation and judging moved to the Gemini API free tier (`LLM_BACKEND=gemini`, `JUDGE_BACKEND=gemini`); Jev still works through OpenRouter (≈ $0.00002 per call). Gemini findings: `gemini-2.5-flash` is closed to new users; the Gemini 3.x Flash models are "thinking" models whose hidden reasoning eats `max_tokens` (fixed with `reasoning_effort: none`); Gemma 4 leaks its thoughts into the answer; RAGAS judging took 128 s for 2 items on 3.8 Flash vs 12.8 s on 3.5 Flash with identical scores.
