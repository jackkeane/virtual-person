# Background Task Queue (RabbitMQ)

Async, background-only job queue for the virtual person: memory curation,
daily-summary precompute, and proactive nudges. It is a sibling of the Redis
seams (`app/infra/redis_client.py`, `app/infra/rate_limit.py`) — a lazy,
fail-soft, **inert-by-default** piece of infrastructure that degrades to a no-op
when its backend is unconfigured or unreachable, and never raises into a caller.

| Piece | File |
|---|---|
| Transport (producer + consumer + fail-soft factory) | `app/queue/task_queue.py` |
| Standalone worker (metrics + graceful shutdown) | `app/queue/worker.py` |
| Job handlers + registry + typed producers | `app/queue/jobs.py` |
| Config (`queue_*` block) | `app/config.py` |
| Live demo | `scripts/demo/queue_demo.sh` |
| Worker entrypoint | `python -m app.queue.worker` |

---

## RED-LINE (interview-critical): never the voice path

**The queue carries BACKGROUND work only. It MUST NEVER sit in the realtime voice
path** (`VAD -> STT -> LLM -> TTS` in `app/ws/handler.py`).

- The **only** app-code enqueue site is the `chat_turn` HTTP endpoint boundary
  (`app/main.py`), *after* the response is computed and `obs_metrics.inc_turn()`
  has run. It is **gated**, wrapped in `try/except`, and **fire-and-forget**
  (a single non-awaited `basic_publish`).
- Enqueue is **not** added to `app/ws/handler.py`, nor to `_run_chat_turn` — the
  function the WebSocket voice path reaches directly. There is **no broker import
  or call anywhere in `app/ws/handler.py`.**
- A broker outage — or the queue being disabled entirely — **changes nothing** for
  a chat turn or a voice turn. The enqueue helper self-gates and swallows all its
  own errors; the worst case is that a background job is simply not scheduled.

```
realtime voice path (app/ws/handler.py):   VAD -> STT -> LLM -> TTS      <- queue NEVER here
background path (app/main.py chat_turn):    ... response ...  --enqueue-->  RabbitMQ
```

The queue is the async, best-effort *back office*; the voice turn is the
synchronous, must-not-fail *front of house*. They never touch.

---

## Architecture

```
  chat_turn HTTP boundary (app/main.py)        <- the ONLY app enqueue site
  typed producers (app/queue/jobs.py) / demo producer
        |
        |  enqueue(type, payload): gated, fire-and-forget, swallows errors
        |  JSON envelope {id, type, payload, attempts, enqueued_at}
        v
  +------------------------------+
  | exchange  vp.tasks           |   direct, durable
  | routing key = vp.task        |
  +--------------+---------------+
                 |
                 v
  +------------------------------+
  | work queue  vp.tasks.q       |   durable; persistent msgs (delivery_mode=2)
  | x-dead-letter-exchange =     |
  |     vp.tasks.dlx             |
  +--------------+---------------+
                 |  basic_qos(prefetch=10) + manual ack
                 v
  +=========================================================================+
  | worker   python -m app.queue.worker                                     |
  |   dispatch {type -> handler}(payload)                                   |
  |                                                                         |
  |   success ------------------------> basic_ack        inc_job_processed  |
  |   handler raised -> attempts += 1                     inc_job_failed    |
  |       attempts <  max_retries(3)                                        |
  |            -> republish to vp.tasks (x-attempts++) ..↺ inc_job_retried  |
  |       attempts >= max_retries(3)                                        |
  |            -> publish to DLX (reason=max-retries)    inc_job_dead_lettered
  |   bad JSON / unknown type -> DLX (decode-error / no-handler)            |
  +===============================+=========================================+
                                  |
                                  v
  +------------------------------+
  | DLX  vp.tasks.dlx            |   fanout, durable
  +--------------+---------------+
                 |
                 v
  +------------------------------+
  | DLQ  vp.tasks.dlq            |   durable; header x-dlq-reason (+ envelope error)
  +------------------------------+   poison messages quarantined for inspection
```

Producer and worker declare this topology **idempotently on connect**, so whoever
connects first creates it and the other converges on the same broker objects.

**Wire envelope** (JSON, UTF-8): `{id, type, payload, attempts, enqueued_at}`.
On retry the same envelope is republished with `attempts` incremented and an
`x-attempts` header. On dead-letter it goes to the DLX with an `x-dlq-reason`
header (`max-retries`, `no-handler`, or `decode-error`); for `max-retries` the
envelope also gains an `error` field.

**Reliability:** durable exchanges/queues + persistent messages + manual ack + QoS
prefetch give at-least-once delivery. Handlers are therefore written to be
**idempotent** where feasible (curation pruning is a no-op once an item is gone;
daily-summary caching is a per-day upsert).

### Retry / dead-letter state machine

With `queue_max_retries = 3` a task that always fails is delivered **3 times**
(attempts `1, 2, 3`) and then parked in the DLQ:

| Delivery | envelope `attempts` in | outcome | ack? |
|---:|---:|---|:---:|
| 1 | 0 | fail -> republish (`x-attempts=1`), `inc_job_retried` | yes |
| 2 | 1 | fail -> republish (`x-attempts=2`), `inc_job_retried` | yes |
| 3 | 2 | fail -> **DLQ** (`reason=max-retries`, `attempts=3`), `inc_job_dead_lettered` | yes |

`max_retries` is the count of **total attempts**, not one try plus three retries.
Non-retryable problems short-circuit to the DLQ on the first delivery: malformed
JSON -> `reason=decode-error`; unknown task `type` -> `reason=no-handler`. Every
delivery is acked (nothing is left unacked to redeliver on reconnect) — except
when a *retry republish itself* fails, in which case the original is `nack`+requeued
so the task is never silently lost.

---

## Inert-by-default gating

Mirrors `app/infra/redis_client.py` + `config._truthy`. The effective gate is:

```
queue_enabled  AND  amqp_url is non-empty  AND  the broker is reachable
```

- `QUEUE_ENABLED` **defaults to `1`** (like `METRICS_ENABLED` / `TTS_CACHE_ENABLED`
  / `RATE_LIMIT_ENABLED`) but the queue stays **dormant** because `AMQP_URL`
  defaults to `""`.
- With `AMQP_URL` unset the app is byte-identical to before: `get_task_queue()`
  returns `None`, `enqueue()` no-ops (returns `False`), **no broker connection is
  opened, and `pika` is never imported**.
- `import pika` happens lazily *inside* the connect/publish methods, so
  `import app.queue.task_queue` (and `app.queue.worker`) works in a pika-less env.
- A failed initial connect is **cached as `None`** so the (already gated) hot path
  never re-pays a connect timeout. `reset_task_queue_cache()` clears the cache
  (used by tests after monkeypatching config).

This is why the default `pytest` suite gains no failures and no new dependency:
the broker-backed tests guard themselves with
`@pytest.mark.skipif(not os.getenv("AMQP_URL"))`. Without a broker the queue suite
is **18 passed, 3 skipped** (the 3 skips are exactly those integration tests).

---

## Jobs

| Task type (`type`) | Handler (`app/queue/jobs.py`) | What it does |
|---|---|---|
| `curate_memories` | `curate_memories` | Re-ranks the memory store, prunes stale items (idempotent). `{scanned, ranked, pruned}` |
| `daily_summary_precompute` | `daily_summary_precompute` | Mirrors `/daily-summary`; upserts a dated `note` cache (idempotent per day). |
| `proactive_nudge` | `proactive_nudge` | `can_send` -> `check_due` -> `mark_sent`; surfaces due reminders. `{sent, reason, due}` |
| `poison` | `poison` | **Always raises** — exists only to exercise the retry/DLQ path. |

Handlers do real work against the same services the HTTP app uses. When `app.main`
is already imported in the process (the same-process case) they reuse the **live**
`MemoryService` / `ProactivityService` singletons via `sys.modules` (importing
nothing themselves); a standalone worker falls back to fresh instances
(`MemoryService` reloads the on-disk store; `ProactivityService` state is
necessarily per-process — which is why `daily_summary_precompute` persists its
result as a dated note).

### Typed producer helpers (fire-and-forget; never raise; return `published?`)

```python
from app.queue import jobs
jobs.enqueue_curate_memories(user_id="")   # -> bool
jobs.enqueue_daily_summary_precompute()    # -> bool
jobs.enqueue_proactive_nudge()             # -> bool
jobs.enqueue_poison()                      # -> bool  (DLQ demo)
```

Each is self-gating: a no-op returning `False` unless the queue is enabled,
configured, and the broker is reachable.

---

## Configuration

All keys live on `AppConfig` in `app/config.py`; every value is overridable via the
env var in parentheses.

| Config attr | Env var | Default | Notes |
|---|---|---|---|
| `queue_enabled` | `QUEUE_ENABLED` | `1` (true) | Master flag; dormant without `amqp_url`. |
| `amqp_url` | `AMQP_URL` | `""` (-> **inert**) | e.g. `amqp://guest:guest@127.0.0.1:5672/` |
| `queue_exchange` | `QUEUE_EXCHANGE` | `vp.tasks` | durable **direct** exchange |
| `queue_name` | `QUEUE_NAME` | `vp.tasks.q` | durable work queue |
| `queue_routing_key` | `QUEUE_ROUTING_KEY` | `vp.task` | bind + publish key |
| `queue_dlx` | `QUEUE_DLX` | `vp.tasks.dlx` | durable **fanout** dead-letter exchange |
| `queue_dlq` | `QUEUE_DLQ` | `vp.tasks.dlq` | durable dead-letter queue |
| `queue_max_retries` | `QUEUE_MAX_RETRIES` | `3` | total attempts before dead-letter |
| `queue_prefetch` | `QUEUE_PREFETCH` | `10` | consumer `basic_qos` prefetch |
| *(worker only)* | `QUEUE_METRICS_PORT` | *(unset)* | If set, the worker serves Prometheus `/metrics` on this port. |

> The env var is `AMQP_URL` and the config attribute is `config.amqp_url`
> (**not** `rabbitmq_url`).

---

## Metrics

One labeled Prometheus family keeps the exposition compact
(`app/observability/metrics.py`):

```
vp_queue_jobs_total{event="enqueued"}        # producer side (app process, chat_turn)
vp_queue_jobs_total{event="processed"}       # worker: handler succeeded, acked
vp_queue_jobs_total{event="failed"}          # worker: handler raised (per attempt)
vp_queue_jobs_total{event="retried"}         # worker: requeued for another attempt
vp_queue_jobs_total{event="dead_lettered"}   # worker: routed to the DLQ
```

- **Producer side:** `inc_job_enqueued()` fires at the `chat_turn` HTTP boundary in
  `app/main.py`, so `enqueued` shows up on the **app's** `/metrics`.
- **Consumer side:** the worker (`app/queue/worker.py`) increments `processed` /
  `failed` / `retried` / `dead_lettered` at each terminal branch of its delivery
  handler, so they show up on the **worker's** `/metrics`.
- The worker binds its own Prometheus endpoint when `QUEUE_METRICS_PORT` is set
  (`_maybe_start_metrics_server` -> `prometheus_client.start_http_server`); it is a
  no-op when the port is unset, keeping the plain worker portless. Metric helpers
  are bound defensively — a missing metrics module degrades every counter to a
  no-op, so instrumentation can never break the worker.
- Labeled counters stay dormant (only the `# HELP`/`# TYPE` header, no series) until
  first incremented — exactly like `vp_tts_cache_total`. In a given process you only
  see the events that process emits (the worker never emits `enqueued`).

> There is also a minimal consumer entrypoint, `python -m app.queue.task_queue`
> (`get_task_queue().consume()`), for a quick manual drain. It does **not** emit
> metrics or bind a port — use `python -m app.queue.worker` for the real thing.

---

## Running it

### Local demo (recommended)

```bash
# 1. a broker (management image so you also get the :15672 UI, optional)
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# 2. pika is optional/lazy — install it for the demo / CI:
~/anaconda3/bin/conda run -n py312 pip install -r requirements.txt

# 3. run the demo
bash scripts/demo/queue_demo.sh
```

The demo (`scripts/demo/queue_demo.sh`) is hermetic — throwaway `MEMORY_PERSIST_PATH`,
every Postgres/Neo4j/Redis DSN unset, so it touches only AMQP. It:

1. **preflights** pika + broker reachability (clear message + non-zero exit if either
   is missing);
2. starts the **real** worker (`app.queue.worker`) in the background, which declares
   the durable topology and serves `/metrics` on `QUEUE_METRICS_PORT` (default `9109`);
3. enqueues a batch — `CURATE_N` `curate_memories` that succeed + one `poison` that
   fails — via the typed producer helpers;
4. waits for the batch to settle, then **shows** the per-outcome counts, the retry
   trail, the work-queue / DLQ **depths** read with a pure-pika passive
   `queue_declare` (`method.message_count` — no management UI required; it *optionally*
   cross-checks the `:15672` API if present), the poisoned message parked in
   `vp.tasks.dlq`, and the live `vp_queue_jobs_total` family scraped from the worker;
5. sends the worker a real **SIGTERM** to drive its own graceful-shutdown path
   (finish the in-flight handler, cancel the consumer, close the AMQP connection).

Useful overrides: `AMQP_URL`, `QUEUE_METRICS_PORT`, `CURATE_N`, `READY_TIMEOUT`,
`SETTLE_TIMEOUT`, `SHUTDOWN_TIMEOUT`, `DEMO_PURGE=0` (keep pre-existing queue
contents), and the whole topology set (`QUEUE_EXCHANGE` / `QUEUE_NAME` /
`QUEUE_ROUTING_KEY` / `QUEUE_DLX` / `QUEUE_DLQ` / `QUEUE_MAX_RETRIES`) to run against
an isolated namespace.

### Standalone worker

```bash
QUEUE_ENABLED=1 AMQP_URL=amqp://guest:guest@127.0.0.1:5672/ QUEUE_METRICS_PORT=9109 \
  ~/anaconda3/bin/conda run -n py312 python -m app.queue.worker
```

Blocks forever, consuming `vp.tasks.q` and dispatching via the `jobs` registry.
Exit codes: `0` when intentionally inert (disabled / no `AMQP_URL`) or after a clean
graceful shutdown; `1` when configured but the broker is unreachable (so a supervisor
restarts it). `SIGINT`/`SIGTERM` trigger a graceful stop.

### CI (`test-rabbitmq` job)

`.github/workflows/ci.yml` runs the suite twice: the bare `test` job (no `AMQP_URL`,
queue inert, broker-backed tests skip) and `test-rabbitmq`, which stands up a
`rabbitmq:3-management-alpine` service and exports:

```bash
QUEUE_ENABLED=1
AMQP_URL=amqp://guest:guest@127.0.0.1:5672/
```

so the `skipif`-gated enqueue/consume + retry/DLQ tests actually execute. `pika` is
pinned in `requirements.txt` (`pika==1.3.2`), so it is present in both jobs; only the
missing `AMQP_URL` keeps the bare job's broker tests dormant.

---

## Honest notes / caveats

- **`pika` is intentionally not installed by default.** It is lazy-imported and
  pinned in `requirements.txt` for CI. Locally, with no broker and no `AMQP_URL`, the
  queue is fully inert and the default test suite is byte-identical to before this
  feature. The demo's preflight fails fast with an actionable message if `pika` (or
  the broker) is missing.
- **`conda run` does not forward OS signals** to the child Python (verified: killing
  the wrapper does not run the child's handler). The demo therefore records the real
  worker PID and sends **`SIGTERM` to the child directly**, exercising the worker's
  own graceful-shutdown handler. Under a real init system (systemd, a container
  `PID 1`) `SIGTERM` reaches the worker directly and runs the *same* path.
- **`ProactivityService` state is per-process (in-memory).** A separate worker does
  not see reminders added over HTTP unless it shares the process. That is deliberate:
  `daily_summary_precompute` bridges the gap by persisting its result as a dated
  `note` memory (a delete-then-write **upsert**, so at most one summary per day —
  redelivery never accumulates notes).
- **At-least-once, not exactly-once.** A crash between "handler succeeded" and "ack"
  causes a redelivery; handlers are written to tolerate it.
- **The worker `/metrics` shows consumer-side events only.** The producer-side
  `enqueued` counter lives in the app process (the `chat_turn` boundary), a
  *different* process from the worker; scrape the app's `/metrics` to see it.
- **`DEMO_PURGE=1` (default) purges `vp.tasks.q` + `vp.tasks.dlq`** at the start for
  deterministic counts. Point the demo at a dedicated/CI broker, or set `DEMO_PURGE=0`,
  if you do not want that.
```
