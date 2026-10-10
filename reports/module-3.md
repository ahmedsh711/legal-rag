# Module 3 report: evaluation, experiment tracking, and the decision layer (JEV-RAG)

**Goal:** turn "it seems to work" into numbers: a golden set, free exact metrics plus an LLM judge, every experiment tracked in MLflow with its data and code versions, and a measured answer to "does a decision model (Jev) or a local reranker make the RAG better, and at what cost?"

## What was built

| Piece | File | Rubric row |
|---|---|---|
| Golden set: 28 AR/EN mirrored pairs (56 questions), 5 categories, checked against the corpus; a separate 5-pair held-out set | `data/golden/golden_set.jsonl`, `data/golden/heldout.jsonl`, `eval/golden.py` | 04 |
| Exact metrics (hit@k, MRR, citations, refusals, language, latency, cost) + offline gate sweep | `eval/metrics.py` | 04 |
| RAGAS 0.4 judge (faithfulness, answer relevancy, context precision/recall; this phase ran faithfulness + context recall), per-item error tolerance, pacing | `eval/ragas_run.py`, `eval/pacing.py` | 04 |
| Judge calibration vs human labels + synthetic negatives (agreement, Cohen's kappa), verbosity probe | `eval/judge.py`, `data/golden/to_label.csv` | 04 |
| Experiment runner: one command per experiment, retrieval-only mode, replay from saved predictions, MLflow logging | `eval/run.py` | 04 |
| MLflow 3.17 server (Postgres + MinIO), runs with lineage tags, config registry with aliases, API served by alias | `docker/compose.yaml`, `eval/track.py`, `config_registry.py` | 04, 05 |
| Decider ABC + Jev (OpenRouter `/systemone`) + local cross-encoder; rerank + answerability gate in the pipeline | `decider/`, `pipeline.py` | 07 |

## How big is the evidence? (read this first)

- **Golden set:** 56 questions = **28 independent pairs** (each asked in Arabic and English; the two halves are translations, not independent samples).
  - 48 answerable = 21 in-scope pairs + 1 explicit-reference pair + 2 repealed pairs. The 6 explicit/repealed questions name their article, so every pipeline variant puts that article first and skips the gate: near-automatic wins that lift hit@1/MRR for all variants equally.
  - 8 unanswerable = 2 off-topic + 2 injection pairs. "100 % correct refusals" therefore rests on **8 questions (4 independent)**.
- **In-sample:** the gate threshold and the choice of decider were made on this same golden set, which was written by the project author (mostly "What is X?" questions whose references paraphrase the gold article). The numbers below are in-sample. A small **held-out check** (5 new pairs, never used for any choice) is reported separately.
- **Noise floor:** the same hybrid configuration scored MRR 0.892 in the retrieval-only run and 0.878 in the end-to-end run (one question ranking differently). Differences of 0.01–0.03 are noise; so is the 0.979 vs 1.000 between hybrid+Jev and dense+Jev (one question).

## Retrieval ablation (retrieval only, no LLM: 48 answerable questions)

| Run | hit@1 | hit@5 | MRR | MRR AR | MRR EN | latency p50 | decider cost / 56 Q |
|---|---|---|---|---|---|---|---|
| hybrid (dense + sparse, RRF) | 0.812 | 0.979 | 0.892 | 0.958 | 0.826 | 0.5 s | $0 |
| **dense only** | **0.938** | 0.979 | **0.955** | 0.951 | **0.958** | 0.7 s | $0 |
| sparse only | 0.604 | 0.833 | 0.692 | 0.810 | 0.573 | 0.7 s | $0 |
| hybrid + local cross-encoder | 0.896 | 0.979 | 0.924 | 0.972 | 0.876 | ~9 s compute (36 s p50 incl. queueing) | $0 |
| hybrid + Jev | 0.979 | 0.979 | 0.979 | 1.000 | 0.958 | 0.6 s | $0.0055 |
| **dense + Jev** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.6 s | $0.0055 |

`hit@5` and `MRR` are measured over the articles shown to the model (`rerank_keep_top` = 5); latencies come from runs at concurrency 4 on a laptop and are indicative only.

Reading it:
- **Dense beat hybrid here.** The sparse (keyword) signal is weak on its own, worst in English (MRR 0.57). Reciprocal-rank fusion gives it the same weight as the dense signal, so on "What is X?" definitions it pulls a neighbouring article above the defining one (936 above 935 for pre-emption, 803 above 802 for ownership, 1033 and 1035 above 1030 for the official mortgage, 121 above 120 for mistake). For Arabic, hybrid and dense are equal. 8 questions changed rank: directionally clear on this question style, not a general law.
- **Jev reranks best** in both languages: one OpenRouter `/systemone` request per question (a 0–3 `score` per article + one `noul` gate, 12 articles in English) costs about **$0.0001 per question** (a 2-article probe cost $0.00002) and ~0.4 s.
- **The local cross-encoder** improves hybrid ranking (0.892 → 0.924) but does not beat dense alone, and needs ~9 s of CPU per question on this laptop. Not viable for serving without a GPU.

**Arabic vs English** (Jev reads English article text): on the 28 pairs, the same top article for all 25 answerable pairs (the 3 differences are off-topic/injection questions, where no article is right) and the same gate decision for 28/28 pairs (mean gate-score difference 0.028, max 0.11). With so few unanswerable pairs the gate agreement is close to trivial; the ranking agreement is the meaningful part. Translating Arabic questions before calling Jev is not needed.

### The gate, replayed offline at every threshold

| Threshold | Local cross-encoder: false / correct refusals | Jev (dense): false / correct refusals |
|---|---|---|
| 0.05 – 0.20 | **0 % / 100 %** | **0 % / 100 %** |
| 0.30 – 0.50 | 6.2 % / 100 % | **0 % / 100 %** |
| 0.75 (≈ the JEV-RAG article's 0.755) | 10.4 % / 100 % | **0 % / 100 %** |
| 0.80 | 12.5 % / 100 % | 2.1 % / 100 % |
| 0.90 | 37.5 % / 100 % | 8.3 % / 100 % |

- Local cross-encoder: off-topic/injection score ≤ 0.011, the lowest answerable question 0.245: a narrow working window (0.05–0.20).
- Jev: off-topic/injection score ≤ 0.04, every answerable question ≥ 0.79: a wide margin. **Chosen gate: 0.5**, the middle of the gap (most room for error on both sides). Chosen on these 8 unanswerable questions, so checked on held-out questions below.
- Repealed-article questions name their article, so the gate never applies to them.
- **A threshold belongs to one model:** ~0.75 is safe for Jev and would wrongly refuse one answerable question in ten with the local model.

### Held-out check (5 new pairs, never used for any choice)

New questions on articles absent from the golden set (651 building collapse, 564 handing over a lease, 199 natural obligations) and two new off-topic questions (passport renewal, a traffic fine), each in Arabic and English. Production config (dense + Jev, gate 0.5), retrieval only:

| Result | Value |
|---|---|
| gold article first (hit@1) | 6 / 6 answerable |
| answerable gate scores | 0.89 – 0.96 (none refused) |
| off-topic gate scores | 0.01 – 0.02 (all 4 refused) |
| Jev cost | $0.001 |

Small (5 pairs), but independent of every tuning decision, and it agrees with the in-sample result.

## End to end (generation + RAGAS judge), Gemini free tier

Generator `gemini-3.1-flash-lite`; RAGAS judge **the same model** (faithfulness + context recall only; answer relevancy and context precision were not run: context precision duplicates the exact hit@k/MRR above and costs one judge call per article on a free daily quota). 56 questions, paced under 12 requests/minute (zero 429s).

| Metric | baseline (hybrid, no decider) | **dense + Jev, gate 0.5** |
|---|---|---|
| hit@1 / MRR | 0.792 / 0.878 | **1.000 / 1.000** |
| citation recall / precision | 0.979 / 0.804 | **1.000** / 0.783 |
| RAGAS context recall | 0.979 | **1.000** |
| RAGAS faithfulness (AR / EN), self-judged | 0.976 (0.986 / 0.965) | 0.969 (0.975 / 0.964) |
| false refusals / correct refusals | 0 % / 100 % | 0 % / 100 % |
| invalid citations, answer language | 0 %, 100 % | 0 %, 100 % |
| decider latency p50 | — | 0.43 s (retrieval 0.24 s) |
| decider cost (56 questions) | $0 | $0.006 |
| LLM tokens per question (prompt / completion) | 1,265 / 64 | 1,087 / 66 |

Reading it:
- **Jev fixes retrieval, not writing.** Every answerable question gets its defining article first and the answers cite it every time; the prompt is ~14 % shorter because the five articles shown are the relevant ones. RAGAS context recall is not an independent signal here: it moves with hit@5.
- **Faithfulness is self-judged and a tie within judge noise.** The judge is the generator's own model (the free tier left no other option: the stronger `gemini-3.5-flash` allows 20 requests/day), so faithfulness is indicative, not independent. Two of the four answers it scored below 0.9 in the Jev run are near-verbatim copies of the gold article (Art. 418 scored 0.75, Art. 935 scored 0.5).
- **Latency is confounded by the free tier.** The whole baseline request had p50 2.0 s; hours later, generation *alone* had p50 5.3 s (p95 25 s) in the Jev run for the same model and similar prompts, while Jev added 0.43 s. Cross-run latency on a shared free tier measures the provider's load; Phase 4's load test measures it properly.
- **Citation precision** (0.804 → 0.783) counts every article the answer names, bracketed or plain; with the defining article first the model more often also names the next article of the same section. All of them were shown to the model (invalid citations stayed 0).

## Judge calibration

**Status: pending.** The calibration run (`python -m legalrag.eval.judge calibrate --mlflow`) needs ~140 judge calls and hit the Gemini Flash-Lite daily cap (500 requests/day) after the experiments above. It will be run with the same judge after the quota resets and this section filled from `reports/eval/judge_calibration.json`.

Ready: 20 answers labelled by a human (all 20 "supported"); because a one-class label set makes kappa meaningless, `judge.py` adds one **synthetic negative** per labelled answer (the same answer plus a fabricated legal claim, label 0 by construction). Caveat: appended fabrications are easier to catch than natural errors, so agreement on them will be optimistic. The self-preference probe is reported as *not measurable* because judge and generator share a model.

## Model registry: the config is served by alias

| Version | Alias | Config | From MLflow run |
|---|---|---|---|
| 1 | `baseline` | no decider, hybrid, gate 0.75 | `e2e-hybrid` |
| 2 | **`production`** | **Jev, dense, gate 0.5** | `e2e-dense-jev` |

- `python -m legalrag.eval.track register --run-id … --alias …` registers exactly the evaluated run's `rag_config.json` and **refuses runs that are not `end_to_end`** (a retrieval-only run never measured an answer).
- The API (`CONFIG_SOURCE=mlflow`) reads `models:/legal-rag-config@production` at startup through two REST calls; `/metadata` shows `"config_source": "legal-rag-config@production (v2)"`, `decider_backend: jev`, `retrieval_mode: dense`, `gate_threshold: 0.5`.
- What the registry controls: the runtime knobs (decider, retrieval mode, top-n, context size, gate). Model identities (LLM, embedding, Jev model) stay in the environment, so "v2 is exactly the evaluated config" holds only if the environment matches. The API checks what it can and **fails startup loudly** on: a value that fails validation (e.g. a typo in `retrieval_mode`), a missing or different `prompt_version`, or a decider model different from the one the gate was tuned for. If the registry is unreachable it keeps its environment settings and logs `config_registry_unreachable` (shown in `/metadata`); a production deployment would alert on it.
- The alias is read at startup only: promoting needs a restart, and replicas restarted at different times can disagree until all have restarted.
- **Rollback demo:** moving `production` to v1 and restarting the API took 18 s (it then served v1: no decider, hybrid); rolling forward to v2 took 22 s. No rebuild, no new image; most of the time is reloading bge-m3.

## Lineage of a run

Every MLflow run records, besides the params (`rag_config`: models, revision, retrieval mode, top-n, context size, decider + model, gate, prompt version, temperature):

| Tag | What it pins |
|---|---|
| `git_sha`, `git_dirty` | the code (short SHA; `git_dirty=true` if the tree had uncommitted changes) |
| `articles_md5`, `index_collection` | the corpus file (the DVC-tracked `articles.json`) and the Qdrant collection built from it |
| `golden_md5` | the question set |
| `uv_lock_md5` | every library version |
| `llm_backend`, `decider_resolved_model` | the endpoint and the exact decider build Jev reported (e.g. `typesafe/jev-1.13-20260917`) |
| `judge_model`, `judge_reasoning_effort`, `ragas_metrics` | how the judge scored |
| `eval_mode` | `retrieval_only` or `end_to_end` |

The retrieval-only runs were re-scored from their saved predictions after the gate-sweep fix (problem 4 below), so their committed `metrics.json` and MLflow runs match the tables above; the stale runs were deleted. Gemini model names like `gemini-3.1-flash-lite` are provider aliases that can change underneath; the date of each run is in MLflow.

## Problems we hit

1. **`typesafe/jev-router` is not the decision model.** It is a chat-model router. Jev decisions use the TypeSafe request body at OpenRouter's `/api/v1/systemone`. A one-call probe confirmed the format before any code.
2. **Dependency chain.** RAGAS 0.4.3 does not import with the newest langchain-community (pinned 0.4.1), and RAGAS → instructor → `jiter<0.15` holds `openai` at 3.3.0 for the whole lock (the API image too). Our code only uses long-stable OpenAI features; to revisit when instructor relaxes the pin.
3. **MLflow global state in tests.** The active experiment id leaked between tests and artifacts were written to `./mlruns` in the repo; each test now gets its own directory and experiment.
4. **The gate sweep counted the run's own gate.** A first sweep reported a flat line because refusals made by the run's 0.75 gate were kept. Predictions now record who refused (`gated` vs the LLM), the sweep replays only the gate, and both rates are stored per threshold.
5. **Out of credits.** OpenRouter answered `402` during the first RAGAS run: the account had never bought credits, and free models are capped at 50 requests/day (one golden run = 56; with RAGAS ≈ 600). Predictions were already saved, so nothing was lost. Generation and judging moved to the Gemini API free tier; Jev still works through OpenRouter. Gemini findings: `gemini-2.5-flash` is closed to new users; the Gemini 3.x Flash models are "thinking" models whose hidden reasoning eats `max_tokens` (fixed with `reasoning_effort: none`); Gemma 4 leaks its thoughts into the answer; RAGAS judging took 128 s for 2 items on 3.8 Flash vs 12.8 s on 3.5 Flash with identical scores.
6. **Free-tier limits.** Flash-Lite allows 15 requests/minute and 500/day; "3.5 Flash" (served as `gemini-3.6-flash`) 5/minute and 20/day. The first Gemini run crashed at question 16 on the per-minute limit → `eval/pacing.py` paces requests under it (zero 429s afterwards); one failed question is now recorded and excluded instead of losing the run.
7. **MLflow 3.17 presigned downloads.** `mlflow.artifacts.download_artifacts` hands the client a presigned URL for `http://minio:9000`, which does not resolve outside Docker; the client retried with back-off for 15 minutes. Config reads stream through the tracking server (`/get-artifact`).
8. **An emoji crashed a Windows console.** MLflow prints "🏃 View run…"; the default cp1252 console cannot encode it. Every CLI switches stdout to UTF-8 in `configure_logging()`.

## Code review and what changed

Two reviewers (Python code; ML-engineering methodology) found no CRITICAL issue.

| Finding | Fix |
|---|---|
| Jev's HTTP client never closed on shutdown | `JevDecider.aclose()`, called by the API's lifespan cleanup (tested) |
| Registry config applied without validation (a typo could silently serve sparse-only) | validated through `Settings`; invalid value / missing prompt version / decider-model mismatch / non-object JSON fail startup loudly (tested) |
| One failed question lost a whole evaluation run | per-item `error`, excluded from metrics and counted (`errors`) |
| Judge verdict parsing crashed on fenced or cut-off JSON; a missing key counted as 0 | tolerant parsing; no usable verdict = left out (not a 0); `unparseable_verdicts` reported |
| Local decider errors returned 500 instead of degrading | mapped to `DeciderUnavailableError` (tested) |
| Odd `cost` value from OpenRouter could lose a good decision | cost parsed safely (tested) |
| Excel label spellings (`1.0`, `TRUE`) crashed label reading | accepted; anything else names the row |
| Committed gate-sweep artifacts predated the sweep fix | retrieval-only runs re-scored from saved predictions; stale MLflow runs deleted |
| Report overstated sample size and independence; judge = generator not stated | "How big is the evidence?" section, held-out check, self-judged label, noise floor |
| Lineage gaps | `git_dirty`, `uv_lock_md5`, `llm_backend`, `decider_resolved_model`, judge settings tags |
| Retrieval-only runs could be promoted | `register` accepts `end_to_end` runs only (tested) |

## Definition of done (Phase 3)

- [x] Golden set + held-out set, exact metrics, RAGAS wrapper, experiment runner with MLflow lineage
- [x] MLflow stack; 9 runs (6 retrieval-only ablations, 2 end-to-end, 1 held-out); registry v1 @baseline, v2 @production; API served by alias; rollback demo
- [x] Decider ABC + Jev + local; pipeline rerank + gate + fallback; per-decider gate thresholds from an offline sweep
- [x] Reviews fixed; walkthrough 03; this report
- [ ] Judge calibration run (waits for the Gemini daily quota)
