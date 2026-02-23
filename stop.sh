#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

APP_PORT="${APP_PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"

stop_pid_file() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "$pid_file" ]]; then
    echo "[stop] no ${name} pid file"
    return 0
  fi

  local pid
  pid="$(cat "$pid_file" || true)"
  if [[ -z "$pid" ]]; then
    echo "[stop] empty ${name} pid file"
    rm -f "$pid_file"
    return 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop] stopping ${name} (PID ${pid})"
    kill "$pid" || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop] force-killing ${name} (PID ${pid})"
      kill -9 "$pid" || true
    fi
  else
    echo "[stop] ${name} process not running (PID ${pid})"
  fi

  rm -f "$pid_file"
}

# App: stop by pid file first, then fallback for manual uvicorn runs.
if [[ -f ".run/app.pid" ]]; then
  stop_pid_file "app" ".run/app.pid"
else
  if curl -fsS "http://127.0.0.1:${APP_PORT}/health" >/dev/null 2>&1; then
    echo "[stop] app pid file missing, searching running uvicorn process on :${APP_PORT}"
    mapfile -t app_pids < <(pgrep -f "uvicorn app.main:app .*--port ${APP_PORT}" || true)

    # Fallback for manual launches without explicit --port arg or different arg order.
    if [[ ${#app_pids[@]} -eq 0 ]]; then
      mapfile -t app_pids < <(pgrep -f "uvicorn app.main:app" || true)
    fi

    if [[ ${#app_pids[@]} -eq 0 ]]; then
      echo "[stop] app process match failed; trying port-owner lookup on :${APP_PORT}"

      # Prefer lsof when available.
      if command -v lsof >/dev/null 2>&1; then
        mapfile -t app_pids < <(lsof -tiTCP:"${APP_PORT}" -sTCP:LISTEN 2>/dev/null || true)
      fi

      # Fallback to ss if lsof is unavailable or returned nothing.
      if [[ ${#app_pids[@]} -eq 0 ]] && command -v ss >/dev/null 2>&1; then
        mapfile -t app_pids < <(ss -ltnp 2>/dev/null | awk -v p=":${APP_PORT}" '$4 ~ p {print $NF}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)
      fi
    fi

    if [[ ${#app_pids[@]} -eq 0 ]]; then
      echo "[stop] app health is up but process was not matched; leaving it running"
    else
      for pid in "${app_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          echo "[stop] stopping app (PID ${pid})"
          kill "$pid" || true
        fi
      done
      sleep 1
      for pid in "${app_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          echo "[stop] force-killing app (PID ${pid})"
          kill -9 "$pid" || true
        fi
      done
    fi
  else
    echo "[stop] no app pid file"
  fi
fi

# Fallback: if vLLM was started manually (no .run/vllm.pid), detect and stop by process match.
if [[ -f ".run/vllm.pid" ]]; then
  stop_pid_file "vllm" ".run/vllm.pid"
else
  # Try to detect a live vLLM API first to avoid false positives.
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[stop] vllm pid file missing, searching running vLLM process on :${VLLM_PORT}"
    mapfile -t vllm_pids < <(pgrep -f "vllm serve .*--port ${VLLM_PORT}" || true)

    # Fallback for commands without explicit --port argument (rare/manual launches)
    if [[ ${#vllm_pids[@]} -eq 0 ]]; then
      mapfile -t vllm_pids < <(pgrep -f "vllm serve" || true)
    fi

    if [[ ${#vllm_pids[@]} -eq 0 ]]; then
      echo "[stop] vllm process match failed; trying port-owner lookup on :${VLLM_PORT}"

      # Prefer lsof when available.
      if command -v lsof >/dev/null 2>&1; then
        mapfile -t vllm_pids < <(lsof -tiTCP:"${VLLM_PORT}" -sTCP:LISTEN 2>/dev/null || true)
      fi

      # Fallback to ss if lsof is unavailable or returned nothing.
      if [[ ${#vllm_pids[@]} -eq 0 ]] && command -v ss >/dev/null 2>&1; then
        mapfile -t vllm_pids < <(ss -ltnp 2>/dev/null | awk -v p=":${VLLM_PORT}" '$4 ~ p {print $NF}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)
      fi
    fi

    if [[ ${#vllm_pids[@]} -eq 0 ]]; then
      echo "[stop] vLLM API is up but process was not matched; leaving it running"
    else
      for pid in "${vllm_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          echo "[stop] stopping vllm (PID ${pid})"
          kill "$pid" || true
        fi
      done
      sleep 1
      for pid in "${vllm_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          echo "[stop] force-killing vllm (PID ${pid})"
          kill -9 "$pid" || true
        fi
      done
    fi
  else
    echo "[stop] no vllm pid file"
  fi
fi

# Extra cleanup for uvicorn --reload mode (reloader parent/child can outlive normal shutdown path).
echo "[stop] stopping extra uvicorn app workers (if any)"
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "python.*uvicorn.*app.main:app" 2>/dev/null || true

# Last-resort app-port cleanup in case process names are unusual.
if command -v lsof >/dev/null 2>&1; then
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop] force-killing app port owner (PID ${pid})"
      kill -9 "$pid" || true
    fi
  done < <(lsof -tiTCP:"${APP_PORT}" -sTCP:LISTEN 2>/dev/null || true)
fi

# CosyVoice worker is launched by app subprocess and may outlive parent on crashes/reloads.
echo "[stop] stopping cosyvoice workers (if any)"
pkill -f "app/voice/cosyvoice_worker.py" 2>/dev/null || true
pkill -f "/anaconda3/envs/cosyvoice/bin/python" 2>/dev/null || true

echo "[stop] done"
