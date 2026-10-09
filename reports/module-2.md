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
| Model weights baked in | no: `hf_cache` volume, or `HF_CACHE` bind to the host cache |
| Machine-specific CA in the image | no: BuildKit secret during build, read-only mount at runtime |

## Problems we hit and what changed

1. **HTTPS inspection** (antivirus) broke downloads in Python and in Docker builds → `truststore` on the host; CA as build secret + runtime mount for containers.
2. **Disk full**: the parity run let FlagEmbedding download the whole model repo (weights twice + 2.2 GB ONNX) → stopped it, deleted only our own caches, parity now loads a pinned local snapshot.
3. **Unpinned weights**: transformers 5 loaded safetensors from a PR branch → `embedding_revision` pinned everywhere.
4. **DVC skipped a stage after code changes**: dependencies were edited while the stage ran; DVC hashes them when the stage finishes → forced rerun; rule written down.
5. **CRLF line endings**: Python on Windows wrote `articles.json` with CRLF, so its md5 (and the collection name) differed from a Linux run → `newline="\n"` in every writer + tests + ruff/pre-commit LF rules.
6. **Index build time varied 3.7×** between two identical runs ({{EMBED_TIMES}}); cause not identified (laptop power/thermal state is the likely suspect, not verified). Length-sorted batching was added to cut padding waste.

## Code review and what changed

{{REVIEW}}

## Definition of done (Phase 2)

{{DOD}}
