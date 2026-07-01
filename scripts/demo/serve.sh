#!/usr/bin/env bash
# Launch the app configured for the Redis + observability demo.
#
# Differs from ./start.sh only in demo-friendly wiring (all overridable via env):
#   - REDIS_URL on DB 15 so sessions / cache / rate-limit go through Redis
#     (DB 15 is a scratch database; the app's real data lives in db0, untouched)
#   - ollama LLM (no vLLM/GPU required) + offline-capable fallback TTS
#   - small rate-limit bucket (capacity 5) so the limiter is easy to trip live
#
# Pair with scripts/demo/run_demo.sh (in a second terminal) to capture numbers.
set -euo pipefail
cd "$(dirname "$0")/../.."

export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/15}"
export SESSION_BACKEND="${SESSION_BACKEND:-auto}"
export TTS_PROVIDER="${TTS_PROVIDER:-fallback}"
export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:32b}"
export RATE_LIMIT_CAPACITY="${RATE_LIMIT_CAPACITY:-5}"
export RATE_LIMIT_REFILL_PER_SEC="${RATE_LIMIT_REFILL_PER_SEC:-0.5}"
export METRICS_ENABLED=1 TTS_CACHE_ENABLED=1 STT_WARMUP_ENABLED=false
PORT="${PORT:-8090}"

echo "[demo] REDIS_URL=$REDIS_URL  LLM=$LLM_PROVIDER  TTS=$TTS_PROVIDER  port=$PORT"
echo "[demo] health: http://127.0.0.1:${PORT}/health   metrics: http://127.0.0.1:${PORT}/metrics"
exec ~/anaconda3/bin/conda run -n py312 uvicorn app.main:app --port "$PORT"
