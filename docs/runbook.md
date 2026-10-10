# Runbook

Every alert links to its section here. Pages (`severity: page`) mean users are affected now; tickets can wait for working hours. Commands assume the repository root and the Compose project `legal-rag`.

## Rolling back

| What changed | Roll back with | Takes |
|---|---|---|
| API release behind nginx (canary) | `uv run python -m legalrag.release rollback` | ~2 s |
| API image | redeploy the previous `sha-<commit>` tag (`API_TAG=sha-abc1234 docker compose ... up -d api`) | ~1 min (model load) |
| Index | `uv run python -m legalrag.index.alias set --alias articles --collection <previous>` (the previous collection is printed on every promotion) | < 1 s |
| System prompt | `uv run python -m legalrag.observability.prompts rollback --version <n>` (only gated versions) | ≤ 60 s (prompt cache) |
| Pipeline config | move the `production` alias of `legal-rag-config` in MLflow to the previous version, restart the API | ~1 min |
| Kubernetes deployment | `kubectl -n legal-rag rollout undo deployment/legal-rag-api` | one rolling update |

## ApiDown

The API stopped answering `/metrics` or `/health`. Check `docker compose ps api` and `docker logs --tail 50 legal-rag-api-1`. Common causes: the index alias is missing (run `dvc repro`), the embedding model is still downloading on a fresh volume, or the container ran out of memory. If the last deploy is the cause, roll back the image.

## AskLatencyHigh

`/ask` p95 is above the 5 s SLO. The Grafana row *RAG behaviour* shows which stage grew: retrieve (see RetrieveStageSlow), decide (Jev latency, check OpenRouter status) or generate (LLM provider or vLLM queue, see LlmQueueBacklog).

## AskErrorsHigh

More than 1 % of `/ask` requests return 5xx. 503 means the LLM provider is down or out of quota (`llm_error` in the logs); 500 means an application bug (`unhandled_error`, search the logs by `request_id`). Roll back if it started with a deploy.

## ErrorBudgetBurnFast

Errors at 14.4 times the allowed rate over both the last hour and the last 5 minutes: at this pace the monthly budget is gone in about two days. Handle as AskErrorsHigh; prefer rolling back the last deploy over debugging live.

## ErrorBudgetBurnSlow

A lower but sustained error rate (6 times the budget over 6 hours). Look at `rag_requests_total` by status over the day and at what changed: deploys (Grafana annotations), provider incidents, a new client.

## RefusalRateHigh

More than half of the answers are refusals. `rag_guardrail_total` by guard tells why: `citation:uncited` means the model stopped citing articles (model or prompt change); `gate:unanswerable` means off-topic traffic or a retrieval problem (check DriftDetected).

## FaithfulnessLow

The nightly online evaluation scored faithfulness below 0.80 on at least 20 judged answers. Compare `prompt_version` and `llm_model` in `rag_info` with the last good evaluation run in MLflow, and read the lowest-scored traces in Langfuse.

## RetrieveStageSlow

Retrieval p95 grew. Check API container CPU (row *Resources*) and that `QUERY_BATCH_MAX` is set: query embedding is the first bottleneck under load. Qdrant latency shows in its own logs.

## LlmQueueBacklog

Requests are waiting in vLLM. If KV-cache usage is low, raise `--max-num-seqs`; if it is high, the GPU is full: shed load (rate limit) or add capacity.

## KvCacheNearlyFull

vLLM's KV cache is above 90 %. Lower `--max-num-seqs` or `--max-model-len`, or move to a larger GPU.

## DailyTokenBudget

Paid tokens over the daily budget. `rag:tokens:1h` by backend shows when it started; look for a single client looping (rate-limit logs) or a prompt that grew.

## RateLimiterDegraded

Redis is unreachable and the rate limiter is letting every request through. Check `docker compose ps redis` and `docker logs legal-rag-redis-1`; requests are still served.

## DriftDetected

The latest drift run found drift. Open the Grafana row *Drift & quality* and the newest report in `reports/drift/`: which features moved, and is it input, retrieval or behaviour drift? Input drift: add examples of the new questions to the golden set before rebuilding anything. Behaviour drift: check what changed in `rag_info`. A rebuild goes through the CT workflow approval and the Airflow `reindex` DAG.

## BatchJobStale

A nightly job (drift check or online evaluation) has not written its metrics for more than 26 hours. Check the Airflow `monitoring` DAG runs, then run the job by hand and read its log:
`uv run --group monitoring python -m legalrag.monitoring.drift_job run --hours 24`.
