# JEV-RAG + best-practice research (web, 2026-10-08)

## What Jev is
- **Jev** = TypeSafe AI's hosted, non-generative "decision model" (launched 2026-09-15, waitlist removed 2026-09-20). It does not write text; it returns typed judgments with calibrated probabilities. Three question types:
  - `noul` — yes/no probability 0–1 (optional `criteria.true` / `criteria.false` descriptions).
  - `choice` — pick one of up to 255 options; returns `choice`, `probabilities` per option, `confidence`.
  - `score` — ordered scale of 2–10 level descriptions; returns probability-weighted `score`, `probabilities` per level, `legend`, `confidence`.
- Latency ~70–500 ms; cost **$0.042 per 1M input tokens, output free**; context 64k per request (32k for `state` + longest question); rate limits ~100K tokens/s, 80 req/s (dynamic). Errors: 401, 422, 429, 529 (backoff). English primary: "other languages, including CJK scripts, are handled but not equally well" (Arabic not mentioned → test it).
- Access: API key at `console.typesafe.ai/keys`, env `TYPESAFE_API_KEY`; `pip install typesafe-sdk`; or through **OpenRouter** (`typesafe/jev-1.13`, `POST https://openrouter.ai/api/v1/systemone` with the OpenRouter key; also `/api/alpha/decisions`). Model ids `jev-1.13.0` (pin), aliases `jev-latest`, `jev-preview`. Docs: `https://docs.typesafe.ai` (`/api`, `/models`, `/llms.txt`), OpenRouter guide `https://openrouter.ai/docs/guides/community/jev`.

Request / response:
```python
from typesafe import TypeSafeClient

client = TypeSafeClient(api_key=...)
r = client.evaluate(
    state="<plain text or structured object/array: question + candidate chunks>",
    model="jev-1.13.0",
    questions={
        "rel_0": {
            "type": "score",
            "instructions": "How relevant is item_0 to the query?",
            "criteria": [
                "Irrelevant",
                "Loosely related",
                "Same subject, different matter",
                "Directly answers",
            ],
        },
        "can_answer": {
            "type": "noul",
            "instructions": "Can the question be fully answered from these items alone?",
            "criteria": {
                "true": "Items directly state the rule/condition",
                "false": "Same topic but the specific point is missing",
            },
        },
        "route": {
            "type": "choice",
            "instructions": "Which book of the code does this question belong to?",
            "criteria": {"book_1": "...", "off_topic": "Not about Egyptian civil law"},
        },
    },
)
r.answers["rel_0"]["score"]
r.answers["can_answer"]["noul"]
r.answers["route"]["choice"]
r.usage
```
POST `/v1/systemone` body = `{model, state, questions}`; response `{model, answers{id: {...}}, usage{input_tokens, output_tokens}}`.

## The JEV-RAG pattern (Gao Dalie, "JEV RAG: A More Efficient Solution for RAG Systems?", 2026-09-27; Thoughtworks "How Jev can help improve the efficiency of RAG pipelines"; MindStudio "JEV as a steerable reranker"; HF blog `hotchpotch/jev-reranker`)
- Idea: most RAG work "isn't thinking, it's deciding". Replace LLM calls for non-generative steps with Jev; keep the LLM only for generation. "Jev does classification, Claude does generation."
- Where Jev goes:
  1. **Ingestion classification** (doc type, sensitive data, entity types) — runs on every document.
  2. **Query routing** before retrieval: which store/filter, needs retrieval at all?, needs rewriting?
  3. **Reranking + sufficiency** after retrieval: one `score` question per chunk, all in one request (parallel), plus a `noul` "can these answer?" gate; retrieve again or decline when low.
  4. **Answer validation**: split the answer into claims, one `noul` per claim against the chunks, flag unsupported content before the user sees it.
- Pattern 1 (used): vector search top-N → Jev rerank (0–3) → Jev answerability → LLM generates with citations. Pattern 2 (PoC): Jev `choice`/`noul` picks metadata tags → WHERE clause → narrowed vector search → rerank → gate → LLM.
- Article's measurements: Jev rerank ≈ Cohere Rerank 3.5 / Amazon Rerank quality at ~1/8 the cost; Jev rerank+answerability in one request ran at 60% of the time and ~1/6 the cost of Cohere + Haiku, with the best ranking. Answerability threshold used: 0.755 on the top chunk; relevance criteria worded so "whether the answer is written is irrelevant" (separates relevance from answer presence). Caveat: recall@5 was already 1.0, so the reranker may be unnecessary when vector search ranks well.
- Thoughtworks tiering: **>0.9 act automatically; 0.6–0.9 ask the LLM for a second opinion; <0.6 escalate / safe default**; keep the probabilities as an audit trail. Start with one high-volume low-risk decision (routing) and run Jev beside the LLM for weeks comparing results. Limitations: early-stage, vendor benchmarks only, hosted (data residency), no explanations, "a badly designed list of choices still produces confident wrong answers".
- MindStudio: steerable reranker — criteria are written policy, so "relevant" can be redefined without retraining; BM25 top-1 21% → 54% with Jev; concurrency 64 → 7.6 s vs 40 s sequential; cost flat vs chunk size while LLM reranking grows linearly; static cross-encoders remain the cheaper default when the relevance definition never changes.
- `jev-reranker` library: `JevReranker(api_key).relevance_rerank(query, docs, threshold=0.2)`; listwise (default) vs pointwise; NanoHotpotQA nDCG@10 hybrid 0.833 → rerank 0.969 → relevance filter 0.975 keeping 7.6 docs; "comparable to bge-reranker-v2-m3" (no formal benchmark). Note: the GitHub repo linked from the article actually uses GPT-4.1-nano structured output for scoring, not Jev — our implementation follows the article's JSON shapes directly.

## Arabic RAG components (arXiv 2506.06339, Naseej Innovation Lab, RAGAS-evaluated on 6 Arabic datasets)
- Chunking: sentence-aware 74.78 avg > fixed 69.41 > recursive 69.13 > semantic 66.92 (semantic chunking consistently worst; fixed-size best only on uniform Wikipedia text).
- Embeddings (overall RAG score): **bge-m3 70.99**, **multilingual-e5-large 70.31**, arctic-embed-l-v2 69.48, gte-multilingual-base 68.48, Arabic-Triplet-Matryoshka-V2 66.46, Arabic-mpnet 45.92 → multilingual contrastive models beat Arabic-specific ones.
- Reranker **bge-reranker-v2-m3**: 70.99 → 74.15 overall; ARCD 80.29 → 86.31 (faithfulness +15); little/no gain on already-clean structured corpora.
- Generation: Aya-8B 72.06 vs StableLM-1.6B 67.08 (bigger, Arabic-capable model matters for faithfulness).
- Eval: RAGAS context precision/recall/faithfulness/answer relevancy; threats: automated judge without human check.

## Embeddings / vector store / serving choices (2026 web)
- bge-m3: MIT, 568M params, 8,192 ctx, 1,024-d, dense + sparse + multi-vector from one model (`FlagEmbedding`); CPU-feasible for ~2.3k texts.
- Qdrant vs pgvector vs Chroma: all fine at this size; Qdrant chosen for native dense+sparse hybrid, payload filters, snapshots and **collection aliases** (atomic index swap = the instructor's "docs_v3 alias swap"). pgvector is the right answer if Postgres already holds the rows; Chroma is simplest but weakest on filters/ops.
- vLLM on RTX 3050 4 GB (github.com/DamnKuldeep/qwen2.5-1.5b-awq-vllm-rtx3050-4gb): `vllm/vllm-openai:v0.11.0` pinned, `Qwen/Qwen2.5-1.5B-Instruct-AWQ` (AWQ + Marlin), weights 1.10 GiB, `--gpu-memory-utilization 0.78` (Windows WDDM reserves VRAM), `--max-num-seqs 32`, chunked prefill; ~22 concurrent chat users within p95 TTFT 1.5 s, 182 tok/s at 20 users, ~300 tok/s plateau, zero preemptions; **set NVIDIA Control Panel → Prefer maximum performance** (default power policy costs 11.6× throughput); thermal throttling not visible from Prometheus under WSL2 → throughput-floor alert instead; CUDA ≥12.8, Docker GPU passthrough via WSL2. We will use `--max-model-len 4096 --max-num-seqs 8` to leave headroom.
- RAGAS: current line 0.4.x (0.4.3, Jan 2026) with `ragas.metrics.collections`; legacy class names deprecated for 1.0 → check import paths via context7. MLflow has `mlflow.genai.scorers.ragas` wrappers.
- Evidently ≥0.7: new `Report` + `Tests` API; embedding drift via `DataDriftPreset(embeddings=..., embeddings_drift_method=...)` or custom `drift_method` (model/domain-classifier default) — verify current API; keep own `drift_stats.py` in scipy as the stable core.
- Langfuse v3 self-host: web + worker + Postgres 17 + ClickHouse + Redis 7 + MinIO; ~512+512+256+1024+128+256 MB baseline, advise 4 vCPU / 8 GB; events queued to S3/MinIO then ingested by the worker; MIT; acquired by ClickHouse (Jan 2026), self-hosting unchanged. Python SDK v3 `@observe`.

## Decisions taken from this research
| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Decider | Jev via OpenRouter/TypeSafe + `LocalDecider` fallback (bge-reranker-v2-m3 + LLM yes/no) | LLM-only reranking; cross-encoder only | JEV-RAG is the user's ask; fallback keeps the project keyless and gives an ablation |
| Embeddings | bge-m3 (dense + sparse) on CPU | multilingual-e5-large (ablation only), Arabic-specific models | paper + hybrid support; GPU reserved for vLLM |
| Chunking | one article (+ paragraph split for long ones) | fixed 512 tokens, semantic | handbook requirement (citations), paper (unit-aware wins) |
| Vector DB | Qdrant | pgvector, Chroma, FAISS | hybrid search, filters, aliases/snapshots |
| Generation | OpenRouter model (default) + vLLM Qwen2.5-1.5B-AWQ (switchable) | local-only | quality for Arabic legal text; vLLM covers handbook boxes |
| Judge | OpenRouter model, pinned, temp 0, calibrated on 20 human labels, per-language reporting | judging with the 1.5B local model | instructor: Arabic judges are weak; small models unreliable |
