# Virtual Person (Ani) — Interview Q&A

Project-focused interview questions and answers, kept in sync with the code on
`main`. Numbers quoted below are from real, reproducible runs (see
[docs/DEMO.md](./docs/DEMO.md), [docs/SEMANTIC_MEMORY.md](./docs/SEMANTIC_MEMORY.md),
[docs/QUEUE.md](./docs/QUEUE.md)).

---

## Part 1 — Project overview

### 1) What problem does this project solve?
**Answer:** It builds an AI companion backend + web client that can chat, remember
user context, manage persona settings, and hold a realtime voice conversation with
a Live2D avatar (VAD → STT → LLM → TTS with viseme lip-sync). Beyond the MVP, it
demonstrates production backend engineering on a real service: Redis-backed
sessions/caching/rate-limiting, Prometheus observability, pgvector semantic memory,
a RabbitMQ background task queue, and a CI + evaluation gate on every push.

### 2) Why FastAPI?
**Answer:** Fast development speed, typed request/response models, first-class
async support (the realtime WS pipeline is asyncio end-to-end), and easy endpoint
organization. It's a good fit for a service that mixes plain HTTP APIs with a
streaming WebSocket voice path.

### 3) What is the system architecture at a high level?
**Answer:**
- **Backend:** FastAPI app — chat, memory, persona, tools, voice, WS realtime pipeline.
- **Client:** Browser UI (Live2D) for text/voice interaction.
- **LLM runtime:** OpenAI-compatible endpoint (vLLM serving `Qwen/Qwen3-14B-AWQ`), with an Ollama fallback behind the same `LLM_PROVIDER` switch.
- **TTS:** CosyVoice2 running as an independent subprocess (own conda env, stdin/stdout JSON protocol).
- **State & infra:** in-memory + JSON-persisted memory with optional PostgreSQL/Neo4j; Redis for sessions/cache/rate-limit; pgvector for semantic recall; RabbitMQ for background jobs.

Three isolated Python runtimes (app / vLLM / TTS worker) talk over HTTP and pipes —
process-level service separation, so a heavy model restart never takes the app down.

### 4) How is memory persistence handled?
**Answer:** Memory items live in process memory and are flushed to a durable JSON
file so restarts don't lose data; PostgreSQL and Neo4j backends are used when
configured. On top of that sits curation: importance scoring, staleness TTL,
dedup, and ranked retrieval.

### 5) How does retrieval work?
**Answer:** Hybrid. The default path is lexical (keyword match over memory fields,
merged across in-memory/Postgres/Neo4j with relevance + recency ranking). With
`SEMANTIC_MEMORY_ENABLED=1`, a semantic layer adds embedding recall — pgvector
(cosine `<=>`, dedicated `vp_mem_vectors` table and DSN) or an in-process numpy
index — and merges it with the keyword results. On paraphrase and cross-lingual
queries the semantic layer took recall from **0/6 to 6/6** (e.g. English "workout
after work" retrieving the Chinese memory "我今天去健身房了", cosine 0.60).

### 6) Which embeddings, and why not put them on the GPU?
**Answer:** Pluggable `EmbeddingService`: a deterministic hash embedder (default —
zero-dependency, CI-safe), Ollama `bge-m3` (1024-dim, multilingual — used for the
real demos), or sentence-transformers. The GPU already holds Qwen3-14B-AWQ plus the
TTS model, so embeddings deliberately run on CPU/ollama instead of competing for
VRAM. Every provider degrades to the hash embedder rather than raising.

### 7) What voice functionality exists?
**Answer:** Full realtime loop: VAD-gated capture → STT (with `zh-CN`/`English`/`Auto`
switching) → streaming LLM → sentence-chunked incremental TTS via an asyncio
producer-consumer queue → WS audio + viseme timeline events that drive Live2D mouth
params with blending/smoothing. Latency instrumentation (`vad_ms`, `stt_ms`, TTFA)
is exported through Prometheus.

### 8) How do you approach safety and observability?
**Answer:** A safety gate with a refusal path plus audit logging, both covered by
the offline eval scorecard (precision/recall/F1 on the gate, so a "never refuse" or
"always refuse" regression fails CI). Observability is Prometheus at `/metrics`
(`vp_turns_total`, `vp_chat_seconds`, cache/rate-limit/queue counters, optional
bearer auth) — every infra feature ships with its own metrics.

### 9) What are the key engineering improvements made recently?
**Answer:** Four backend-infrastructure features, each implemented → adversarially
reviewed → verified with real numbers → gated in CI:
1. **Redis + observability** — sessions (7-day TTL), TTS response cache (hit: ~2.2 s → ~3 ms, ≥99.7% cut), per-IP Lua token-bucket rate limiting (burst 10 → 5 allowed / 5 rejected), Prometheus `/metrics`.
2. **Evaluation + CI** — deterministic offline scorecard + optional LLM-as-judge; GitHub Actions matrix of **5 jobs** (default, Redis, RabbitMQ, pgvector service containers, eval artifact).
3. **Semantic memory** — pgvector + bge-m3 hybrid retrieval, paraphrase/cross-lingual recall 0/6 → 6/6.
4. **RabbitMQ task queue** — durable exchange → work queue → DLX/DLQ, manual ack, retries ≤3, publisher confirms + `mandatory`; strictly outside the realtime voice path.

### 10) What would you improve next?
**Answer:** LLM tracing (LangFuse) on top of the Prometheus layer; a dedicated
vector DB (Milvus) once memory volume justifies it; debounced/sampled queue jobs
instead of enqueue-per-turn; per-item memory edit UX; a self-hosted GPU runner so
CI can also gate true voice-path latency (cloud CI is CPU-only, so latency budgets
are currently enforced locally, not in CI — documented honestly).

---

## Part 2 — Backend infrastructure deep-dive

### B1) Everything is "gated" — what does that mean and why?
**Answer:** Every infra feature is **inert by default**: without its env var
(`REDIS_URL`, `SEMANTIC_MEMORY_ENABLED`, `AMQP_URL`) the code path is byte-identical
to the baseline — no import of the driver, no connection attempt, no behavior
change. Enabled but unreachable ⇒ fail-soft degrade (cache miss, allow-all
rate-limit, keyword-only retrieval, dropped enqueue), never a 500. That's why the
default CI job stays green with zero services while three other jobs exercise the
same suite against real Redis/RabbitMQ/pgvector containers. Pattern: a module-level
cached factory (`get_*() -> client | None`) with short timeouts that caches `None`
on failure.

### B2) Why is the message queue *not* in the realtime voice path?
**Answer:** The whole voice design is about minimizing TTFA (time-to-first-audio) —
filler audio, incremental sentence-chunked TTS, echo holdoff. Adding a broker
round-trip inside a sub-second loop would be an architectural mistake. The boundary:
**in-process asyncio** for realtime work (STT/LLM/TTS streaming, the existing
`tts_queue` producer-consumer), **RabbitMQ** for deferrable background work (memory
curation/dedup/staleness, daily summaries, embedding backfill). Enqueue happens
only in the HTTP `chat_turn` wrapper — never in `_run_chat_turn`, which the WS
voice path shares — and `app/ws/handler.py` has zero broker references (enforced
by review + grep).

### B3) Walk me through the queue's failure semantics.
**Answer:** Durable direct exchange `vp.tasks` → durable work queue `vp.tasks.q`
with a dead-letter exchange → `vp.tasks.dlq`. Messages are persistent
(`delivery_mode=2`); the worker uses manual ack with QoS prefetch and acks **only
after** the handler succeeds. On failure it republishes with an `x-attempts` header
up to 3 retries, then dead-letters with a reason header. The producer uses publisher
confirms **plus `mandatory=True`** — confirms alone ack unroutable messages, so
without `mandatory` a topology mistake silently drops work. Demo run: 3 jobs
processed, a poison job failed 3 times, retried twice, and landed in the DLQ with
`x-dlq-reason: max-retries`.

### B4) What concurrency bug did you hit with pika?
**Answer:** `pika.BlockingConnection`/`BlockingChannel` is **not thread-safe**, and
FastAPI executes sync `def` endpoints on a threadpool — so a single shared channel
could be used from multiple threads concurrently, corrupting the AMQP framing.
Fix: serialize the enqueue critical section with a `threading.Lock`, reset
channel/connection on error inside the lock, and use double-checked locking in the
cached factory so concurrent first calls can't build two connections. Found in
adversarial review before it ever hit the broker, together with three more real
bugs: missing `mandatory` (silent message loss), a worker exit-code bug, and a
DLQ-publish-failure path that requeued in a hot loop.

### B5) The TTS cache claims a 99.7% cut — what's the mechanism?
**Answer:** TTS synthesis is the expensive tail of a voice turn (~1–2 s). The cache
keys Redis on `(provider, text)` and stores the synthesized waveform, so a hit
replaces synthesis with a single Redis `GET` — measured **2231 ms → 2.9 ms (763×)**
on one run, 1069 ms → 2.6 ms (407×) on another; the stable claim is ≥99.7%. It
composes with sentence-chunked TTS: recurring sentences ("让我想想…") hit the cache
even inside novel replies.

### B6) Why token bucket for rate limiting, and why key on client IP?
**Answer:** Token bucket permits short bursts while capping sustained rate — right
for conversational traffic. It runs as a single atomic Redis **Lua** script (read,
refill, consume in one step — no check-then-act race across app threads/workers),
keyed on **client IP** rather than the client-supplied `user_id`, which would be
trivially spoofable. Fail-open on Redis outage: rate limiting is protection, not a
feature users should see fail. Verified: burst of 10 → 5 allowed, 5 rejected (429).

### B7) How do you evaluate quality, and how does CI gate it?
**Answer:** Two layers. (1) A deterministic, no-LLM offline scorecard
(`eval/run_eval.py`): safety-gate precision/recall/F1 on labeled inputs/outputs +
memory-retrieval top-1 — thresholds that fail CI on regression, requiring precision
*and* recall so degenerate always/never-refuse gates can't pass. (2) An optional
LLM-as-judge harness scoring live answers on relevance/groundedness/persona; with
vLLM `Qwen/Qwen3-14B-AWQ` as the system under test and an **independent** judge
model (ollama `qwen3:32b`), the live stack scored **4.69/5** overall. CI runs 5
jobs on every push: default (all infra dormant) + Redis + RabbitMQ + pgvector
service containers + the eval artifact job. Suite: **201 passed / 5 skipped**
default; the broker-gated tests (21) run in the RabbitMQ job.

### B8) What did building this teach you about CI environments?
**Answer:** Two hard lessons. First, my dev env had extra packages, so 7 async
tests silently **skipped** in a clean CI env until `pytest-asyncio` was pinned —
the suite looked green while testing less; CI parity means a venv built from
`requirements.txt` alone. Second, `python -m pytest` adds the cwd to `sys.path`
but bare `pytest` doesn't — CI's bare invocation couldn't import `app` until
`pythonpath` was set in `pyproject.toml`. Verification has to replicate CI's
*exact* command in a clean environment.

---

## Part 3 — Quick factual Q&A

### Q1) Does the project use FastAPI?
**Answer:** Yes — the backend API and the WS realtime pipeline are FastAPI.

### Q2) How is embedding done?
**Answer:** Via a pluggable `EmbeddingService` (hash / ollama `bge-m3` 1024-dim /
sentence-transformers), feeding pgvector or an in-process index. Gated: with
`SEMANTIC_MEMORY_ENABLED` unset, retrieval is pure keyword and behavior is
byte-identical to the pre-semantic baseline.

### Q3) Is retrieval hybrid?
**Answer:** Yes, in two senses: hybrid **sources** (in-memory + Postgres + Neo4j,
merged with relevance/recency ranking) and hybrid **method** (lexical + embedding
recall merged when the semantic layer is enabled). No cross-encoder reranker yet —
candidate volume doesn't justify one; noted as future work.

### Q4) Does lip-sync use visemes?
**Answer:** Yes. The server emits viseme timeline events; the client maps them to
Cubism mouth params (`ParamMouthOpenY`, `ParamMouthForm`) with a layered
expression/speech/micro-motion blend and smoothing.

### Q5) Which Live2D assets?
**Answer:** Default Hiyori assets for now; an original avatar set is on the roadmap.

---

## Part 4 — Latency Q&A

### L1) What are the main latency bottlenecks?
**Answer:** LLM generation, TTS synthesis (especially cold start), and end-to-end
streaming overhead. Measured per stage via the `vad_ms` / `stt_ms` / TTFA metrics;
a real vLLM chat turn is ~0.6–4.5 s of inference, while the deterministic persona
fast-path answers in ~2 ms — which is why the demos distinguish the two explicitly.

### L2) What did you do to reduce latency?
**Answer:** Streaming LLM → sentence chunking → incremental TTS (first audio plays
while later sentences still synthesize); filler audio to mask dead air; TTS response
caching (hit ≈ 3 ms); component warmup; conservative GPU allocation
(`--voice-profile`) to avoid VRAM contention between the 14B LLM and TTS; blocking
work offloaded from the event loop (`run_in_executor`).

### L3) How would you present latency in an interview?
**Answer:** Stage-by-stage: capture → STT → LLM first-token/total → TTS
first-audio/total → playback start, reporting cold vs warm paths separately, since
warm latency is what users actually experience. The Prometheus histograms
(`vp_chat_seconds` etc.) make this a dashboard query rather than an anecdote.
