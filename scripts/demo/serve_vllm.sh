#!/usr/bin/env bash
# Launch the app on the PRODUCTION LLM stack: vLLM serving Qwen/Qwen3-14B-AWQ
# over its OpenAI-compatible API on :8000 (LLM_PROVIDER=openai_compat), plus the
# same Redis + observability wiring as serve.sh and the semantic-memory layer
# ENABLED (bge-m3 embeddings via ollama, in-process cosine index).
#
# One server backs all three vLLM demos:
#   - scripts/demo/run_demo.sh            (Redis/metrics + a REAL LLM turn)
#   - scripts/demo/semantic_demo_vllm.sh  (semantic recall -> vLLM generation)
#   - scripts/demo/run_llm_eval_vllm.sh   (LLM-judge over real Qwen replies)
#
# Prereqs: vLLM up on :8000 (project start.sh / your venv), Redis up, ollama up
# with bge-m3 pulled (for embeddings).
#
# Hermetic by default: routes memory at a throwaway persist file and clears the
# Postgres/Neo4j DSNs so demo seeds never touch your real memory stores. Redis
# uses scratch DB 15 (db0 untouched). Everything is overridable via env.
set -euo pipefail
cd "$(dirname "$0")/../.."

# --- production LLM: vLLM / Qwen3-14B-AWQ over the OpenAI-compatible API ---
export LLM_PROVIDER="${LLM_PROVIDER:-openai_compat}"
export OPENAI_COMPAT_BASE_URL="${OPENAI_COMPAT_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_COMPAT_MODEL="${OPENAI_COMPAT_MODEL:-Qwen/Qwen3-14B-AWQ}"

# --- Redis + observability (same knobs as serve.sh) ---
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/15}"
export SESSION_BACKEND="${SESSION_BACKEND:-auto}"
export TTS_PROVIDER="${TTS_PROVIDER:-fallback}"
export RATE_LIMIT_CAPACITY="${RATE_LIMIT_CAPACITY:-5}"
export RATE_LIMIT_REFILL_PER_SEC="${RATE_LIMIT_REFILL_PER_SEC:-0.5}"
export METRICS_ENABLED=1 TTS_CACHE_ENABLED=1 STT_WARMUP_ENABLED=false

# --- semantic memory: bge-m3 embeddings via ollama, in-process cosine index ---
# (no PGVECTOR_DSN -> NumpyVectorIndex fallback; this is what lets a chat turn
#  recall a paraphrased memory and feed it to the LLM.)
export SEMANTIC_MEMORY_ENABLED="${SEMANTIC_MEMORY_ENABLED:-1}"
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ollama}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge-m3}"
export OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-bge-m3}"

# --- hermetic memory: throwaway store, never touch real Postgres/Neo4j ---
export MEMORY_PERSIST_PATH="${MEMORY_PERSIST_PATH:-${TMPDIR:-/tmp}/vp-demo-vllm-mem.json}"
unset MEMORY_POSTGRES_DSN MEMORY_NEO4J_URI MEMORY_NEO4J_USERNAME MEMORY_NEO4J_PASSWORD 2>/dev/null || true

PORT="${PORT:-8090}"
echo "[demo/vllm] LLM=openai_compat -> $OPENAI_COMPAT_MODEL @ $OPENAI_COMPAT_BASE_URL"
echo "[demo/vllm] REDIS_URL=$REDIS_URL  SEMANTIC=$SEMANTIC_MEMORY_ENABLED (emb=$EMBEDDING_PROVIDER/$EMBEDDING_MODEL)  port=$PORT"
echo "[demo/vllm] health: http://127.0.0.1:${PORT}/health   metrics: http://127.0.0.1:${PORT}/metrics"

# preflight (warn only): is vLLM actually up? The app still boots either way and
# surfaces LLM errors per-turn, so this is a hint, not a hard gate.
if ! curl -sf -m 3 "${OPENAI_COMPAT_BASE_URL%/}/models" >/dev/null 2>&1; then
  echo "[demo/vllm] WARNING: vLLM not reachable at ${OPENAI_COMPAT_BASE_URL%/}/models — start it first (project start.sh / your venv)."
fi

exec ~/anaconda3/bin/conda run --no-capture-output -n py312 uvicorn app.main:app --port "$PORT"
