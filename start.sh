#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="dev"
WITH_VLLM="false"
VOICE_PROFILE="false"
APP_PORT="8080"
VLLM_PORT="8000"

usage() {
  cat <<'EOF'
Usage: ./start.sh [--dev|--prod] [--with-vllm] [--voice-profile] [--app-port N] [--vllm-port N]

Options:
  --dev            Run uvicorn with --reload (default)
  --prod           Run uvicorn without --reload
  --with-vllm      Start vLLM if port is not already serving
  --voice-profile  Start vLLM in low-VRAM voice mode (2048 ctx, lower GPU util)
  --app-port N     App port (default: 8080)
  --vllm-port N    vLLM port (default: 8000)
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) MODE="dev"; shift ;;
    --prod) MODE="prod"; shift ;;
    --with-vllm) WITH_VLLM="true"; shift ;;
    --voice-profile) VOICE_PROFILE="true"; WITH_VLLM="true"; shift ;;
    --app-port) APP_PORT="${2:-}"; shift 2 ;;
    --vllm-port) VLLM_PORT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

for cmd in curl; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Missing required command: $cmd"; exit 1; }
done

mkdir -p logs .run

export LLM_PROVIDER="${LLM_PROVIDER:-openai_compat}"
export OPENAI_COMPAT_BASE_URL="${OPENAI_COMPAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
export OPENAI_COMPAT_MODEL="${OPENAI_COMPAT_MODEL:-Qwen/Qwen3-14B-AWQ}"
export OPENAI_COMPAT_API_KEY="${OPENAI_COMPAT_API_KEY:-dummy}"
export WS_STREAM_ENABLED="${WS_STREAM_ENABLED:-true}"
export VOICE_SPEED_MODE="${VOICE_SPEED_MODE:-fast}"
export STT_LANGUAGE_HINT="${STT_LANGUAGE_HINT:-zh}"
export STT_WARMUP_ENABLED="${STT_WARMUP_ENABLED:-true}"
export STT_DEVICE="${STT_DEVICE:-cpu}"

if [[ "$VOICE_PROFILE" == "true" ]]; then
  VLLM_CMD="vllm serve Qwen/Qwen3-14B-AWQ --port ${VLLM_PORT} --enforce-eager --max-model-len 2048 --gpu-memory-utilization 0.55 --max-num-seqs 2"
else
  VLLM_CMD="vllm serve Qwen/Qwen3-14B-AWQ --port ${VLLM_PORT} --enforce-eager --max-model-len 4096 --gpu-memory-utilization 0.55"
fi

if [[ "$WITH_VLLM" == "true" ]]; then
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[start] vLLM already running on :${VLLM_PORT}"
  else
    if [[ "$VOICE_PROFILE" == "true" ]]; then
      echo "[start] starting vLLM on :${VLLM_PORT} with low-VRAM voice profile..."
    else
      echo "[start] starting vLLM on :${VLLM_PORT}..."
    fi

    nohup bash -lc "cd ~ && source .venv/bin/activate && ${VLLM_CMD}" \
      > logs/vllm.log 2>&1 &
    echo $! > .run/vllm.pid

    echo -n "[start] waiting for vLLM"
    for _ in {1..120}; do
      if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo " - ready"
        break
      fi
      echo -n "."
      sleep 1
    done
    echo
  fi
else
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[start] detected running vLLM on :${VLLM_PORT}"
  else
    echo "[start] vLLM not detected on :${VLLM_PORT} (OK if using ollama/other provider)"
  fi
fi

if [[ "$MODE" == "dev" ]]; then
  UVICORN_ARGS=(--reload)
else
  UVICORN_ARGS=()
fi

if [[ -f .run/app.pid ]]; then
  old_pid="$(cat .run/app.pid || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[start] app already running with PID ${old_pid}. Use ./stop.sh first."
    exit 1
  fi
fi

echo "[start] launching app on :${APP_PORT} (${MODE})"
nohup ~/anaconda3/bin/conda run -n py312 uvicorn app.main:app --port "$APP_PORT" "${UVICORN_ARGS[@]}" \
  > logs/app.log 2>&1 &
echo $! > .run/app.pid

APP_HEALTH="http://127.0.0.1:${APP_PORT}/health"
echo -n "[start] waiting for app"
for _ in {1..60}; do
  if curl -fsS "$APP_HEALTH" >/dev/null 2>&1; then
    echo " - ready"
    break
  fi
  echo -n "."
  sleep 1
done
echo

echo "[start] Done"
echo "  App:        http://127.0.0.1:${APP_PORT}"
echo "  Web client: http://127.0.0.1:${APP_PORT}/client/"
echo "  Health:     ${APP_HEALTH}"
echo "  Logs:       ${ROOT_DIR}/logs/app.log"
if [[ "$WITH_VLLM" == "true" ]]; then
  echo "  vLLM logs:  ${ROOT_DIR}/logs/vllm.log"
  if [[ "$VOICE_PROFILE" == "true" ]]; then
    echo "  vLLM mode:  low-VRAM voice profile"
  fi
fi
