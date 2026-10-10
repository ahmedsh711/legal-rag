#!/usr/bin/env bash
# Alert drill: break things on purpose and check that the right alert reaches the right channel.
#   bash monitoring/alert_drill.sh            (monitoring + core + llm profiles running)
# Timeline (minutes): 0 start a trickle of traffic (3 users, 45 min) | 5 stop Redis | 9 start
# Redis | 14 stop the API | 17 start the API. Expected deliveries in monitoring/alerts/alerts.jsonl:
#   RateLimiterDegraded (ticket, cause)  ~2-3 min after Redis stops, resolved after it is back
#   ApiDown (page, symptom)              ~1-2 min after the API stops, resolved after it is back
#   RefusalRateHigh (ticket, symptom)    ~30 min in: the small vLLM model's answers are refused
set -euo pipefail
OUT=${OUT:-monitoring/alerts/drill.log}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT"; }
pause() { sleep "$(( $1 * 60 ))"; }

log "start traffic (3 users, 45 min)"
LOCUST_STEPS="" uv run --group load python -m locust -f loadtest/locustfile.py --headless \
  -u 3 -r 1 -t 45m --host http://127.0.0.1:8010 --only-summary > monitoring/alerts/drill-locust.log 2>&1 &
pause 5;  log "stop redis";  docker stop legal-rag-redis-1 > /dev/null
pause 4;  log "start redis"; docker start legal-rag-redis-1 > /dev/null
pause 5;  log "stop api";    docker stop legal-rag-api-1 > /dev/null
pause 3;  log "start api";   docker start legal-rag-api-1 > /dev/null
wait
log "traffic done"
