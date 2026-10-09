# Module 2 report: vanilla RAG behind a FastAPI service in Docker

**Goal:** answer an Arabic or English question about the Egyptian Civil Code over HTTP, with the article numbers the answer is based on, from a container a reviewer can start with three commands. This is the baseline every later phase (Jev decisions, guardrails, evaluation) is measured against.

## Result

| Item | Value |
|---|---|
| Index | Qdrant collection `{{COLLECTION}}` behind alias `articles`, {{POINTS}} points (1,093 live articles × AR + EN, + 56 repeal notes, − 1: Art. 1022 has no Arabic text) |
| Embedding | bge-m3 dense (1024-d) + sparse, CPU, pinned revision `9a0624b8`, max 1,024 tokens per text |
| Index build time (embedding only, laptop CPU) | {{EMBED_TIMES}} |
| bge-m3 parity vs official FlagEmbedding | {{PARITY}} |
| Retrieval spot check (10 concepts × AR/EN, paraphrased questions) | {{SPOT}} |
| Query latency, retrieval only (embed + Qdrant hybrid) | {{RETRIEVE_MS}} |
| `/ask` end to end (OpenRouter `qwen/qwen3-235b-a22b-2507`) | {{ASK}} |
| Tests / coverage | {{TESTS}} |
| API image `ahmedshobaki/legal-rag-api` | {{IMAGE}} |

## How a question becomes an answer

```
POST /ask ─▶ RequestIdMiddleware (id, timing, safe 500)
          ─▶ AskRequest validation (3–2000 chars, Arabic or Latin words) ── bad ─▶ 422
          ─▶ RagPipeline.ask
               ├─ Retriever.retrieve  (in a worker thread: embedding is CPU work)
               │    ├─ explicit "Article N" / "المادة N" references first (repealed included)
               │    └─ hybrid search: dense + sparse prefetch, RRF fusion in Qdrant, is_repealed = false
               ├─ nothing found ─▶ fixed refusal, no LLM call
               ├─ Generator.complete (OpenAI-compatible: OpenRouter or vLLM, timeout + 2 retries)
               └─ cited_articles → sources; citations never shown to the model → invalid_citations
          ─▶ AskResponse {answer, sources, refused, invalid_citations, request_id, prompt_version, model, usage, timings_ms}
LLM provider down / timeout ─▶ 503 + Retry-After (their outage, not our bug)
```

## Index: decisions that make it safe to rebuild

| Decision | Why | Rejected alternative |
|---|---|---|
| One point per article *per language* (+ one "note" point for repealed articles) | Arabic questions match Arabic text best, English questions English text; both points carry the same payload, results are merged per article | One point with both texts concatenated: one vector has to mean two languages at once |
| Header prepended to the embedded text (`المادة N - section / topic`) | Questions name the topic ("sale", "البيع") that the article text itself often does not repeat | Raw text only |
| Collection name = hash of (articles.json md5, model, revision, max length, normalization version) | Same inputs → same collection; new inputs never overwrite what is serving | One fixed collection rebuilt in place (downtime, no rollback) |
| Alias `articles` moved atomically after the build | Zero-downtime deploy and one-call rollback | Restarting the API with a new collection name |
| `index_meta` side collection (model, revision, normalization, source md5, git SHA) + API refuses a mismatch at startup | The instructor's incident: embedding model changed in a rebuild, the app kept the old one, 18% of Arabic answers silently degraded | Trusting the configuration |
| bge-m3 with plain `transformers` + parity script | The official package adds six heavy libraries to the image for ~40 lines of math | FlagEmbedding in the image |
| Pinned model revision | transformers 5 silently fetched a safetensors conversion from a PR branch; a pin makes "which weights?" answerable | `main` (moves when the model owner pushes) |

Rollback demo: {{ROLLBACK}}

## Retrieval spot check

Ten concepts with a known answer article, asked once in Arabic and once in English with wording that does not copy the article (a preview of the Phase 3 golden set, not a benchmark):

{{SPOT_TABLE}}

## API behaviour

| Case | Status | Body |
|---|---|---|
| Valid question | 200 | answer + sources + request id |
| `question` too short / only symbols / unknown field | 422 | `{"detail": [{"field": ..., "message": ...}]}` |
| Index unreachable or empty | `/health` 503 | `{"status": "unavailable"}` |
| LLM provider error or timeout | 503 + `Retry-After: 10` | `{"detail": "LLM provider unavailable", "request_id": ...}` |
| Anything unexpected | 500 | `{"detail": "Internal server error", "request_id": ...}`, no traceback |
| Index built with another model / revision / normalization | API refuses to start | `IndexMismatchError` in the log |

## Docker

| Check | Result |
|---|---|
| Multi-stage (uv builder → slim runtime with only the venv) | yes |
| CPU-only torch (no CUDA wheels) | yes, `[tool.uv.sources]` → pytorch-cpu index |
| Runs as non-root `app` (uid 1000) | {{NONROOT}} |
| HEALTHCHECK on `/health`, 180 s start period | {{HEALTH}} |
| Model weights baked in | no: downloaded once into the `hf_cache` named volume |
| Machine-specific CA in the image | no: BuildKit secret during build, read-only mount at runtime |

## Problems we hit and what changed

1. **HTTPS inspection** (antivirus) broke downloads in Python and in Docker builds → `truststore` on the host; CA as build secret + runtime mount for containers.
2. **Disk full, twice.** First the parity run let FlagEmbedding download the whole model repo (weights twice + a 2.2 GB ONNX copy) → stopped it, deleted only our own caches, parity now loads a pinned local snapshot. Then Docker's virtual disk grew during an image rebuild (C: down to 8 MB) → removed our old image and build cache inside Docker (that does not shrink the virtual disk), and with the owner's permission purged the pip download cache (3.4 GB) and temp files older than 7 days (5.3 GB). A bind mount of the host's Hugging Face cache (to avoid a second 2.3 GB download) was tried and removed: inside the container a Windows bind mount is `root:root 755`, so the non-root `app` user cannot write to it; the `hf_cache` named volume is used.
3. **Unpinned weights**: transformers 5 loaded safetensors from a PR branch → `embedding_revision` pinned everywhere.
4. **DVC skipped a stage after code changes**: dependencies were edited while the stage ran; DVC hashes them when the stage finishes → forced rerun; rule written down.
5. **CRLF line endings**: Python on Windows wrote `articles.json` with CRLF, so its md5 (and the collection name) differed from a Linux run → `newline="\n"` in every writer + tests + ruff `line-ending = "lf"`. (A pre-commit `mixed-line-ending` hook was tried and removed: the antivirus held its freshly created `.exe` launcher indefinitely; `.gitattributes eol=lf` already keeps checkouts LF.)
6. **Index build time varied 3.7×** between two identical unsorted runs; cause not identified (the Windows power scheme was "Silent" during a later run; not verified for the earlier ones). Length-sorted batching cut the next build to 611 s.
7. **Parity "failed" because the reference was set up wrong**: dense matched to 1e-6 but sparse did not. FlagEmbedding only loads its trained sparse head when `colbert_linear.pt` sits next to `sparse_linear.pt`; otherwise it silently uses a random head (INFO-level log). After adding the file: PARITY OK.
8. **Qdrant client 1.19 vs server 1.16** (outside the supported one-minor-version gap) → server pinned to v1.19.2; the volume was recreated and the index rebuilt rather than migrated.
9. **DVC could not see the real output**: on a fresh clone or an empty Qdrant, `dvc repro` said "up to date" while the index was missing → `always_changed: true` on the index stage; the build reuses an identical finished collection in seconds without loading the model.

## Code review and what changed

Three reviewers (python, fastapi, security) read the code; none found a CRITICAL issue. Every HIGH and MEDIUM finding was fixed test-first unless a ruling below says why not.

| Finding | Severity | Fix |
|---|---|---|
| Concurrent requests share one fast tokenizer (not thread-safe) | HIGH | `threading.Lock` around `encode` (one query embedding at a time; CPU-bound anyway) |
| `dvc repro -f` would delete and refill the collection that is serving | HIGH | `finished_build()` → reuse a finished identical collection; `TEXT_FORMAT_VERSION` in the hash so changed text never reuses old vectors |
| Failure mid-stream: client sees a cut stream, no error | HIGH | SSE `{"type": "error", ...}` event; test with a failing stream |
| Volume mount points owned by root, app runs as uid 1000 | HIGH | Dockerfile creates `/app/feedback` and the HF cache dir owned by `app` |
| `/health` called the sync Qdrant client on the event loop | HIGH | `asyncio.to_thread` |
| Services published on 0.0.0.0 with dev passwords / no auth | HIGH | all ports on `127.0.0.1`; Postgres/MinIO passwords required (`${VAR:?}`) |
| Every provider error was a 503 "retry later", including 401/404 | MEDIUM | 503 + Retry-After only for connection errors, timeouts, 429, 5xx; other 4xx → 502 |
| Stream not closed on disconnect; request coroutine created before iteration | MEDIUM | lazy request + `finally: await stream.close()` |
| Empty corpus or duplicate article ids could go live | MEDIUM | `ValueError` on empty input; stored-vs-sent point count check before the alias moves |
| "particle 5" / "Article 12345" matched as article references | MEDIUM | word boundary before `article`, no digit after the number |
| Context cut to 5 could drop articles the user named | MEDIUM | keep `max(context_size, explicit references)` |
| Blocking file write in async `/feedback`; free-form `request_id` | MEDIUM | plain `def` endpoint (threadpool); `request_id` must match the id pattern |
| CA placeholder ignored by `*.pem` → fresh clone cannot build | MEDIUM | `!docker/no-local-ca.pem`; entrypoint checks `-f` before `-s` |
| No Qdrant readiness before the API starts | MEDIUM | Qdrant healthcheck (bash `/dev/tcp`) + `service_healthy` |
| Empty `choices`, `assert` as a runtime check, middleware reset not in `finally`, `.env.*` not ignored, no lifespan cleanup, no `X-Accel-Buffering` | LOW | all fixed |

Rulings (not changed, with the cost if wrong):
- HEALTHCHECK stays on `/health` (readiness), not `/live`: Docker does not restart unhealthy containers, so "unhealthy" honestly means "cannot serve"; Kubernetes gets separate probes in Phase 6. Cost if wrong: low.
- The api service keeps `env_file: ../.env`: every service is local and loopback-only; per-service secrets arrive with Kubernetes Secrets in Phase 6. Cost if wrong: low.
- No auth or throttling on `/ask` and `/feedback` yet: Redis token bucket is Phase 4; ports are loopback-only. Cost if wrong: low.
- Pinned revision lives in `params.yaml`, `IndexParams` and `Settings`: the API refuses to start on any drift. Cost if wrong: none.
- Base images pinned by tag, not digest: course scale. Cost if wrong: low.

## Definition of done (Phase 2)

{{DOD}}
