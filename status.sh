#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

APP_PORT="${APP_PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"

if [[ $# -ge 2 ]]; then
  case "$1" in
    --app-port) APP_PORT="$2"; shift 2 ;;
  esac
fi
if [[ $# -ge 2 ]]; then
  case "$1" in
    --vllm-port) VLLM_PORT="$2"; shift 2 ;;
  esac
fi

print_proc_status() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "$pid_file" ]]; then
    echo "${name}: not tracked (no pid file)"
    return
  fi

  local pid
  pid="$(cat "$pid_file" || true)"
  if [[ -z "$pid" ]]; then
    echo "${name}: pid file empty"
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "${name}: running (PID ${pid})"
  else
    echo "${name}: stale pid file (PID ${pid} not running)"
  fi
}

check_http() {
  local name="$1"
  local url="$2"
  if curl -fsS "$url" >/dev/null 2>&1; then
    echo "${name}: healthy (${url})"
  else
    echo "${name}: unreachable (${url})"
  fi
}

echo "== Process status =="
print_proc_status "app" ".run/app.pid"
print_proc_status "vllm" ".run/vllm.pid"

echo
echo "== Health checks =="
check_http "app" "http://127.0.0.1:${APP_PORT}/health"
check_http "vllm" "http://127.0.0.1:${VLLM_PORT}/v1/models"

echo
echo "== Port listeners =="
if command -v ss >/dev/null 2>&1; then
  ss -ltnp 2>/dev/null | awk -v a=":${APP_PORT}" -v v=":${VLLM_PORT}" '
    NR==1 || index($4, a) || index($4, v) {print}
  '
else
  echo "ss not found; skipping listener report"
fi

echo
echo "== Logs =="
[[ -f logs/app.log ]] && echo "app log:  $ROOT_DIR/logs/app.log" || echo "app log:  missing"
[[ -f logs/vllm.log ]] && echo "vllm log: $ROOT_DIR/logs/vllm.log" || echo "vllm log: missing"
