# Redis + Observability — Live Demo

This walks through **Feature 3**: four pieces of backend infrastructure added to
the realtime voice pipeline — a Redis session store, Prometheus metrics, per-caller
rate limiting, and a TTS response cache — and captures the real numbers from a
live run so they can be reproduced, not just claimed.

## Design principle: safe by default, gated on `REDIS_URL`

Every Redis-backed path is inert unless `REDIS_URL` is set **and** Redis is
reachable (see `app/infra/redis_client.py`). With it unset the app behaves
exactly as before Feature 3 — in-memory sessions, no cache, allow-all rate limit
— so the default single-process deploy and the whole test suite (`141 passed`)
are byte-identical to baseline. Turning the features on is a matter of pointing
`REDIS_URL` at a Redis instance; nothing else changes.

This mirrors the pre-existing `app/memory/service.py` Postgres/Neo4j gating: the
same "degrade to a working fallback, never hard-depend on infra" pattern.

## What runs where

| Component | Runtime | Notes |
|---|---|---|
| App (uvicorn) | conda env `py312`, port 8090 | `scripts/demo/serve.sh` |
| Redis | local `:6379`, **DB 15** | scratch DB; app's real data in db0 is never touched |
| LLM | ollama `qwen3:32b` | no vLLM/GPU needed for the demo |
| TTS | fallback provider | gTTS when online, offline sine WAV otherwise |

## Reproduce

```bash
# terminal 1 — start the app wired for the demo (Redis DB 15, ollama, fallback TTS)
scripts/demo/serve.sh

# terminal 2 — drive traffic and print the numbers
scripts/demo/run_demo.sh
```

`run_demo.sh` flushes **DB 15** for a clean slate (never db0), then exercises all
four features and reads the results back from `/metrics` and `redis-cli`.

## Captured results

Headline numbers from one live run (`docs/demo/CAPTURE.txt`, full `/metrics` in
`docs/demo/metrics_snapshot.txt`):

| Capability | Evidence | Number |
|---|---|---|
| **TTS response cache** | miss = full synth, hit = one Redis `GET` | hit collapses synth to a **~3 ms** GET — **≥99.7% latency cut** (this run **2231 ms → 2.9 ms, 763×**) |
| **Per-caller rate limiting** | burst of 10 rapid requests, one client IP, capacity 5 | **5 allowed, 5 rejected**; `vp_rate_limited_total 0 → 5` |
| **Redis session store** | conversation history persisted per user | `vp:sess:alice` holds the turn list, **TTL 604800 s (7 days)** |
| **HTTP metrics** | `/chat/turn` now moves Prometheus counters | `vp_turns_total 0 → 3`, `vp_chat_seconds_count 3` (flat before this change) |

> The cache "miss" latency tracks the synth provider (here gTTS over the network,
> ~2 s and variable); the "hit" is a constant ~3 ms Redis `GET`. The stable,
> meaningful number is the hit: it removes essentially all of the synth cost.

### Raw run

```text
========== SECTION 1 — HTTP metrics + Redis session store ==========
vp_turns_total (HTTP path): 0.0 -> 3.0   (was flat before this change)
chat-latency samples recorded: 3.0
session key(s) in Redis:  vp:sess:alice
alice conversation history (Redis LIST): 6 entries (3 user + 3 assistant), each {role, content, at}
alice session TTL: 604800 s  (7-day bound -> idle keys evaporate)

========== SECTION 2 — TTS response cache: miss (full synth) vs hit (Redis GET) ==========
  miss=2.026770s  hit=0.002557s
  miss=2.152225s  hit=0.002904s
  miss=2.546725s  hit=0.003784s
  miss=2.199118s  hit=0.002454s
  --> avg MISS=2231.2 ms | avg HIT=2.92 ms | 763x faster | 99.87% latency cut
  cache counter delta this run: miss +4, hit +4

========== SECTION 3 — per-caller rate limiting (keyed on client IP, capacity=5) ==========
burst of 10 rapid requests from one client IP -> allowed=5, rejected=5
vp_rate_limited_total: 0.0 -> 5.0
bucket key: vp:rl:127.0.0.1  TTL=60 s  (keyed on caller IP, not the spoofable user_id)
```

### Redis keys created (DB 15)

```
vp:sess:alice          # session history (RPUSH + LTRIM + 7-day TTL)
vp:sess:burst
vp:rl:127.0.0.1        # token-bucket state, keyed on caller IP (atomic Lua, 60 s TTL)
vp:tts:3c65ac36...     # cached waveform, key = sha256(provider|text), 24 h TTL
vp:tts:4010cf7c...
vp:tts:547322ac...
vp:tts:949768b9...
```

## Hardening (review follow-ups)

Four low-severity findings from the Feature 3 review, since fixed:

- **Rate-limit identity** — the limiter now keys on the caller's **client IP**
  (`request.client.host` / `websocket.client.host`), not the client-supplied
  `user_id` a caller could rotate to dodge the limit. Sessions still key on
  `user_id` (that's the correct per-user scope). Behind a proxy, prefer
  `X-Forwarded-For`.
- **Rate-limit ordering** — on the WS voice path the gate now runs **before STT**,
  so a flood is rejected without paying for transcription (single token per turn,
  no double-consume).
- **Filler audio off the event loop** — the "thinking…" filler lookup/synth is
  offloaded via `run_in_executor`, so a cold filler cache can't block the async
  loop.
- **`/metrics` auth** — set `METRICS_AUTH_TOKEN` to require
  `Authorization: Bearer <token>` on `/metrics`; unset leaves it open (dev default).

## Honest notes

- **The cache "miss" is a real synthesis** through the fallback TTS provider; the
  "hit" is a single Redis `GET` (~3 ms). Same behavior in front of
  CosyVoice/FishAudio — the cache wraps any provider identically.
- **`vp_tts_cache_total{result="miss"}` starts non-zero** because the app
  pre-warms filler audio ("嗯…", "让我想想…") into the cache at startup; the demo
  reports the per-run *delta* (miss +4 / hit +4) to isolate its own traffic.
- **Voice-path histograms** (`vp_vad_seconds`, `vp_stt_seconds`,
  `vp_ttfa_seconds`, `vp_tts_seconds`) read 0 here because this demo drives the
  **HTTP** path; they populate on the WebSocket voice path (`app/ws/handler.py`),
  which emits VAD/STT/TTFA/TTS latency and the turn counter.
- **DB 15** is used throughout so the demo is self-contained and disposable; the
  app's live data (db0) is never read or written.
