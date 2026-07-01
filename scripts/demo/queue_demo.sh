#!/usr/bin/env bash
# Background task-queue live demo for virtual-person (Feature: async background jobs).
#
# Proves the whole producer -> RabbitMQ -> worker -> ack path end to end, plus the
# failure path (retry -> dead-letter), against a REAL broker and the REAL worker
# (`app.queue.worker`) — the numbers an interviewer would ask about:
#   1. a batch enqueues and the worker processes + ACKs every delivery
#   2. a poisoned job retries up to QUEUE_MAX_RETRIES times, then dead-letters
#   3. the DLQ holds exactly that message (attempts=max_retries, reason=max-retries)
#   4. the worker's own /metrics exposes the vp_queue_jobs_total family, populated
#      live by the consumer (processed / failed / retried / dead_lettered)
#   5. a real SIGTERM drives the worker's graceful-shutdown path (not a kill -9)
#
# RED-LINE (interview-critical): this queue is BACKGROUND-ONLY. It NEVER sits in
# the realtime voice path (VAD->STT->LLM->TTS in app/ws/handler.py). This demo
# talks to the queue via the typed producer helpers and consumes with the real
# standalone worker; it never touches the voice path. See docs/QUEUE.md.
#
# Prereqs:
#   * a RabbitMQ broker reachable at $AMQP_URL, e.g.:
#       docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
#   * pika installed in the py312 conda env (it is intentionally optional/lazy —
#     the default test suite stays broker-free — so CI/this demo installs it via
#     `pip install -r requirements.txt`). The preflight checks both and exits with
#     a clear, actionable message if either is missing.
#
# Safe + hermetic: the worker runs with a throwaway MEMORY_PERSIST_PATH and with
# every Postgres/Neo4j/Redis DSN unset, so it never touches the real memory store
# or any external datastore. Only AMQP (the vp.tasks.* topology) is used.
#
# Python is invoked ONLY through the py312 conda env, per project policy.
set -uo pipefail

# --------------------------------------------------------------------------- #
# Config (all overridable from the environment)
# --------------------------------------------------------------------------- #
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AMQP_URL="${AMQP_URL:-amqp://guest:guest@127.0.0.1:5672/}"
QUEUE_METRICS_PORT="${QUEUE_METRICS_PORT:-9109}"
CURATE_N="${CURATE_N:-3}"              # curate_memories jobs that succeed
READY_TIMEOUT="${READY_TIMEOUT:-25}"   # max seconds to wait for the worker /metrics
SETTLE_TIMEOUT="${SETTLE_TIMEOUT:-25}" # max seconds to wait for the batch to settle
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-10}"  # max seconds to wait for graceful stop
DEMO_PURGE="${DEMO_PURGE:-1}"          # purge work queue + DLQ first for clean counts
RABBITMQ_MGMT_URL="${RABBITMQ_MGMT_URL:-http://127.0.0.1:15672}"

# Queue topology (defaults MUST match app/config.py; override the whole set
# together to run against an isolated namespace without touching vp.tasks.*).
QUEUE_EXCHANGE="${QUEUE_EXCHANGE:-vp.tasks}"
QUEUE_NAME="${QUEUE_NAME:-vp.tasks.q}"
QUEUE_ROUTING_KEY="${QUEUE_ROUTING_KEY:-vp.task}"
QUEUE_DLX="${QUEUE_DLX:-vp.tasks.dlx}"
QUEUE_DLQ="${QUEUE_DLQ:-vp.tasks.dlq}"
QUEUE_MAX_RETRIES="${QUEUE_MAX_RETRIES:-3}"

STATE="$(mktemp -d)"
export DEMO_STATE_DIR="$STATE"

# Hermetic env for every python child: queue on, throwaway memory, no datastores.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export QUEUE_ENABLED=1
export AMQP_URL QUEUE_METRICS_PORT CURATE_N
export QUEUE_EXCHANGE QUEUE_NAME QUEUE_ROUTING_KEY QUEUE_DLX QUEUE_DLQ QUEUE_MAX_RETRIES
export MEMORY_PERSIST_PATH="$STATE/memory_store.json"
unset MEMORY_POSTGRES_DSN MEMORY_NEO4J_URI MEMORY_NEO4J_USERNAME MEMORY_NEO4J_PASSWORD PGVECTOR_DSN REDIS_URL 2>/dev/null || true

WPID=""        # the REAL python worker PID (from the pid file, not the conda wrapper)
CONDA_WPID=""  # the `conda run` wrapper PID (backgrounded job to reap)

PY() { "$HOME/anaconda3/bin/conda" run --no-capture-output -n py312 python "$@"; }
hr() { printf '\n========== %s ==========\n' "$1"; }
note() { printf '  %s\n' "$1"; }
scrape() { curl -s "http://127.0.0.1:$QUEUE_METRICS_PORT/metrics" 2>/dev/null; }
# metric_val "<metrics snapshot>" EVENT -> integer count for that event (0 if absent)
metric_val() {
  printf '%s\n' "$1" | awk -v pat="vp_queue_jobs_total{event=\"$2\"}" \
    '$1==pat{print int($2); f=1} END{if(!f) print 0}'
}

cleanup() {
  # Graceful first: SIGTERM the REAL python child (conda run does NOT forward
  # signals to it, so we target the child PID recorded in the pid file). Fall back
  # to SIGKILL only if it ignores the graceful stop. Then reap the conda wrapper.
  if [ -n "${WPID:-}" ] && kill -0 "$WPID" 2>/dev/null; then
    kill -TERM "$WPID" 2>/dev/null || true
    for _ in $(seq 1 $((SHUTDOWN_TIMEOUT * 5))); do kill -0 "$WPID" 2>/dev/null || break; sleep 0.2; done
    kill -KILL "$WPID" 2>/dev/null || true
  fi
  [ -n "${CONDA_WPID:-}" ] && wait "$CONDA_WPID" 2>/dev/null
  [ -n "${STATE:-}" ] && rm -rf "$STATE" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for() { # wait_for FILE TIMEOUT_SECONDS -> 0 if it appears, 1 on timeout
  local f="$1" t="${2:-30}" i=0
  while [ ! -e "$f" ]; do
    i=$((i + 1)); [ "$i" -ge $((t * 5)) ] && return 1
    sleep 0.2
  done
}

# --------------------------------------------------------------------------- #
# Inline helpers (written to the throwaway state dir; run in the py312 env)
# --------------------------------------------------------------------------- #
# AMQP utility: pure-pika passive inspection (no management UI required).
cat >"$STATE/amqp.py" <<'PY'
import json, os, sys
import pika  # lazy dep; preflight guarantees it is installed

URL = os.environ["AMQP_URL"]
WORKQ = os.environ.get("QUEUE_NAME", "vp.tasks.q")
DLQ = os.environ.get("QUEUE_DLQ", "vp.tasks.dlq")


def _conn(timeout=5.0):
    p = pika.URLParameters(URL)
    p.socket_timeout = timeout
    p.connection_attempts = 1
    p.retry_delay = 0
    return pika.BlockingConnection(p)


def _depth(conn, q):
    # A passive declare that 404s closes the channel, so use a fresh one per queue.
    ch = conn.channel()
    try:
        return ch.queue_declare(queue=q, passive=True).method.message_count
    except Exception:
        return -1
    finally:
        try:
            ch.close()
        except Exception:
            pass


def cmd_ping():
    _conn(3.0).close()
    print("broker-ok")


def cmd_purge():
    conn = _conn()
    ch = conn.channel()
    for q in (WORKQ, DLQ):
        try:
            ch.queue_declare(queue=q, passive=True)
            ch.queue_purge(q)
        except Exception:
            # Not declared yet (fresh broker) -> nothing to purge; reopen channel.
            try:
                ch.close()
            except Exception:
                pass
            ch = conn.channel()
    conn.close()
    print("purged")


def cmd_depths():
    conn = _conn()
    print(f"{WORKQ} {_depth(conn, WORKQ)}")
    print(f"{DLQ} {_depth(conn, DLQ)}")
    conn.close()


def cmd_peekdlq():
    conn = _conn()
    ch = conn.channel()
    method, props, body = ch.basic_get(DLQ, auto_ack=False)
    if method is None:
        print("DLQ empty")
        conn.close()
        return
    try:
        env = json.loads(body)
    except Exception:
        env = {"_raw": (body[:200] if isinstance(body, (bytes, bytearray)) else b"").decode("utf-8", "replace")}
    hdr = props.headers or {}
    print(json.dumps({
        "id": env.get("id"),
        "type": env.get("type"),
        "attempts": env.get("attempts"),
        "error": env.get("error"),
        "x-dlq-reason": hdr.get("x-dlq-reason"),
    }, ensure_ascii=False))
    # Preserve it in the DLQ so the depth reading above stays honest.
    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    conn.close()


if __name__ == "__main__":
    {"ping": cmd_ping, "purge": cmd_purge, "depths": cmd_depths, "peekdlq": cmd_peekdlq}[sys.argv[1]]()
PY

# Producer: enqueue a batch via the REAL typed producer helpers (fire-and-forget).
cat >"$STATE/producer.py" <<'PY'
import os
from app.queue import jobs  # typed, self-gating, never-raise producer helpers

n = int(os.environ.get("CURATE_N", "3"))
ok = sum(1 for i in range(n) if jobs.enqueue_curate_memories(f"demo-user-{i}"))
poison = jobs.enqueue_poison()
print(f"curate_enqueued={ok}/{n} poison_enqueued={int(bool(poison))}")
PY

# Worker: a THIN wrapper around the REAL standalone worker entrypoint
# (app.queue.worker.main). main() already: declares the durable topology, starts
# the Prometheus /metrics server on QUEUE_METRICS_PORT, consumes with manual ack +
# retry/DLQ, emits the vp_queue_jobs_total counters, and shuts down gracefully on
# SIGINT/SIGTERM. The ONLY thing this shim adds is recording the child's PID: the
# demo sends SIGTERM to it directly because `conda run` does not forward signals.
cat >"$STATE/worker.py" <<'PY'
import os, pathlib, sys
from app.queue.worker import main

pathlib.Path(os.environ["DEMO_STATE_DIR"], "worker.pid").write_text(f"{os.getpid()}\n")
sys.exit(main())
PY

# --------------------------------------------------------------------------- #
# 0) Preflight — pika installed + broker reachable
# --------------------------------------------------------------------------- #
hr "preflight: py312 env + pika + broker"
note "project root : $ROOT"
note "AMQP_URL     : $AMQP_URL"
note "topology     : exchange=$QUEUE_EXCHANGE (direct) key=$QUEUE_ROUTING_KEY -> queue=$QUEUE_NAME"
note "             : DLX=$QUEUE_DLX (fanout) -> DLQ=$QUEUE_DLQ  |  max_retries=$QUEUE_MAX_RETRIES"
note "state dir    : $STATE  (throwaway; removed on exit)"

if ! PY -c "import pika" 2>/dev/null; then
  printf '\nFAIL: pika is not installed in the py312 conda env.\n'
  printf '      It is intentionally optional (lazy import) so the default test\n'
  printf '      suite stays broker-free. Install it for this demo / in CI:\n\n'
  printf '        ~/anaconda3/bin/conda run -n py312 pip install -r requirements.txt\n\n'
  exit 1
fi
note "pika         : installed"

if ! PY "$STATE/amqp.py" ping >/dev/null 2>&1; then
  printf '\nFAIL: no RabbitMQ broker reachable at %s\n' "$AMQP_URL"
  printf '      Start one and re-run:\n\n'
  printf '        docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management\n\n'
  exit 1
fi
note "broker       : reachable"

# --------------------------------------------------------------------------- #
# 1) Start the REAL worker (background) — declares topology + serves /metrics
# --------------------------------------------------------------------------- #
hr "start worker (app.queue.worker, background)"
PY "$STATE/worker.py" >"$STATE/worker.log" 2>&1 &
CONDA_WPID=$!
if ! wait_for "$STATE/worker.pid" 15; then
  printf 'FAIL: worker process did not start. Log:\n'; sed 's/^/  | /' "$STATE/worker.log" 2>/dev/null
  exit 1
fi
WPID="$(tr -d '[:space:]' <"$STATE/worker.pid")"

# Ready == the worker's /metrics endpoint answers. main() starts it only AFTER a
# successful broker connect, so a live endpoint proves the worker is consuming.
READY=0
for _ in $(seq 1 $((READY_TIMEOUT * 5))); do
  if scrape | grep -q '^vp_queue_jobs_total\|^# TYPE vp_queue_jobs_total'; then READY=1; break; fi
  kill -0 "$WPID" 2>/dev/null || break   # worker exited early (e.g. broker vanished)
  sleep 0.2
done
if [ "$READY" != "1" ]; then
  printf 'FAIL: worker did not become ready (no /metrics on :%s). Log:\n' "$QUEUE_METRICS_PORT"
  sed 's/^/  | /' "$STATE/worker.log" 2>/dev/null
  exit 1
fi
note "worker PID   : $WPID   (real python child; conda-run wrapper=$CONDA_WPID)"
note "metrics      : http://127.0.0.1:$QUEUE_METRICS_PORT/metrics"
grep -oE 'event=worker_start .*' "$STATE/worker.log" | head -1 | sed 's/^/  worker: /' || true

# Deterministic counts: clear any residue from a prior run (topology already declared).
if [ "$DEMO_PURGE" != "0" ]; then
  PY "$STATE/amqp.py" purge >/dev/null 2>&1 && note "purged       : $QUEUE_NAME + $QUEUE_DLQ (clean slate)"
fi

# --------------------------------------------------------------------------- #
# 2) Enqueue a batch: several curate_memories (succeed) + one poison (fails)
# --------------------------------------------------------------------------- #
hr "enqueue batch: ${CURATE_N}x curate_memories (ok) + 1x poison (fails)"
ENQ="$(PY "$STATE/producer.py")"
note "$ENQ"

# --------------------------------------------------------------------------- #
# 3) Wait for the worker to settle the batch (successes acked; poison DLQ'd)
# --------------------------------------------------------------------------- #
hr "drain — worker processes + acks; poison retries then dead-letters"
SETTLED=0
SNAP=""   # last /metrics scrape (guards `set -u` if SETTLE_TIMEOUT<1 skips the loop)
for _ in $(seq 1 $((SETTLE_TIMEOUT * 5))); do
  SNAP="$(scrape)"
  P="$(metric_val "$SNAP" processed)"; D="$(metric_val "$SNAP" dead_lettered)"
  { [ "$P" -ge "$CURATE_N" ] && [ "$D" -ge 1 ]; } && { SETTLED=1; break; }
  sleep 0.2
done
[ "$SETTLED" = "1" ] || note "(warning: batch did not fully settle within ${SETTLE_TIMEOUT}s; showing current state)"
note "processed(ok)=$(metric_val "$SNAP" processed)  failed=$(metric_val "$SNAP" failed)  retried=$(metric_val "$SNAP" retried)  dead_lettered=$(metric_val "$SNAP" dead_lettered)"
printf '  per-delivery worker log (event=...):\n'
grep -oE 'event=(job_ok|job_retry|dlq)[^\n]*' "$STATE/worker.log" | sed 's/^/    /' \
  || note "    (no per-delivery lines captured)"

# --------------------------------------------------------------------------- #
# 4) Inspect queues via pure AMQP passive declare (no management UI needed)
# --------------------------------------------------------------------------- #
hr "queue depths (pika passive queue_declare -> method.message_count)"
PY "$STATE/amqp.py" depths | while read -r q d; do
  printf '  %-14s ready=%s\n' "$q" "$d"
done
printf '  poison in DLQ (basic_get peek; nacked back so the depth stays honest):\n'
PY "$STATE/amqp.py" peekdlq | sed 's/^/    /'

# Optional: cross-check via the management HTTP API if :15672 is up (guest/guest).
MGMT_HOST="$(printf '%s' "$RABBITMQ_MGMT_URL" | sed -E 's#^https?://##; s#/.*##; s#:.*##')"
MGMT_PORT="$(printf '%s' "$RABBITMQ_MGMT_URL" | sed -E 's#^https?://[^:/]+:?##; s#/.*##')"; MGMT_PORT="${MGMT_PORT:-15672}"
if timeout 2 bash -c "exec 3<>/dev/tcp/${MGMT_HOST}/${MGMT_PORT}" 2>/dev/null; then
  dlq_msgs="$(curl -s -u guest:guest "$RABBITMQ_MGMT_URL/api/queues/%2F/$QUEUE_DLQ" | grep -oE '"messages":[0-9]+' | head -1)"
  [ -n "$dlq_msgs" ] && note "mgmt API (:$MGMT_PORT) $QUEUE_DLQ -> ${dlq_msgs}"
fi

# --------------------------------------------------------------------------- #
# 5) Scrape the worker's /metrics — the vp_queue_jobs_total family (populated)
# --------------------------------------------------------------------------- #
hr "worker /metrics — vp_queue_jobs_total family (scrape :$QUEUE_METRICS_PORT)"
SNAP_FILE="$ROOT/docs/demo/queue_metrics_snapshot.txt"
mkdir -p "$(dirname "$SNAP_FILE")"
scrape >"$SNAP_FILE"
grep -E '^# (HELP|TYPE) vp_queue_jobs_total|^vp_queue_jobs_total\{' "$SNAP_FILE" | sed 's/^/  /' \
  || note "(no vp_queue_* family found — is app.observability.metrics importable?)"
note "breakdown    : processed=success, failed=handler-raised, retried=requeued, dead_lettered=DLQ"
note "full snapshot saved -> docs/demo/queue_metrics_snapshot.txt"
note "note: the 'enqueued' series is emitted in the APP process at the chat_turn"
note "      HTTP boundary (app/main.py), a different process from this worker."

# --------------------------------------------------------------------------- #
# 6) Graceful shutdown — real SIGTERM to the worker (its own signal handler)
# --------------------------------------------------------------------------- #
hr "graceful shutdown (SIGTERM -> worker's signal handler)"
kill -TERM "$WPID" 2>/dev/null || true
GRACEFUL=0
for _ in $(seq 1 $((SHUTDOWN_TIMEOUT * 5))); do kill -0 "$WPID" 2>/dev/null || { GRACEFUL=1; break; }; sleep 0.2; done
wait "$CONDA_WPID" 2>/dev/null
CONDA_WPID=""; WPID=""   # reaped: let cleanup() skip the kill path
grep -oE 'event=(signal|worker_stop) .*' "$STATE/worker.log" | sed 's/^/  worker: /' \
  || note "(worker stopped)"
[ "$GRACEFUL" = "1" ] && note "shutdown     : clean (finished in-flight work, cancelled consumer, closed AMQP)" \
                       || note "shutdown     : forced (did not exit within ${SHUTDOWN_TIMEOUT}s)"

hr "summary"
note "producer -> $QUEUE_EXCHANGE (direct, key=$QUEUE_ROUTING_KEY) -> $QUEUE_NAME -> worker: every delivery acked."
note "poison failed ${QUEUE_MAX_RETRIES}x (attempts=1..${QUEUE_MAX_RETRIES}) -> dead-lettered via $QUEUE_DLX (fanout) -> $QUEUE_DLQ."
note "the worker's /metrics moved live: processed=${CURATE_N}, failed=${QUEUE_MAX_RETRIES}, retried=$((QUEUE_MAX_RETRIES - 1)), dead_lettered=1."
note "a broker outage here changes NOTHING for a chat or voice turn — enqueue is gated + fire-and-forget."
note "see docs/QUEUE.md for the architecture, the RED-LINE, and the inert-by-default gating."
