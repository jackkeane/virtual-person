# Redis + Observability — Live Demo

This walks through **Feature 3**: four pieces of backend infrastructure added to
the realtime voice pipeline — a Redis session store, Prometheus metrics, per-user
rate limiting, and a TTS response cache — and captures the real numbers from a
live run so they can be reproduced, not just claimed.

## Design principle: safe by default, gated on `REDIS_URL`

Every Redis-backed path is inert unless `REDIS_URL` is set **and** Redis is
reachable (see `app/infra/redis_client.py`). With it unset the app behaves
exactly as before Feature 3 — in-memory sessions, no cache, allow-all rate limit
— so the default single-process deploy and the whole test suite (`139 passed`)
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
| **TTS response cache** | miss = full synth, hit = one Redis `GET` | avg **1069 ms → 2.62 ms** = **407× faster, 99.75% latency cut** (4/4 phrases) |
| **Per-user rate limiting** | burst of 10 rapid requests, one user, capacity 5 | **5 allowed, 5 rejected**; `vp_rate_limited_total 0 → 5` |
| **Redis session store** | conversation history persisted per user | `vp:sess:alice` holds the turn list, **TTL 604800 s (7 days)** |
| **HTTP metrics** | `/chat/turn` now moves Prometheus counters | `vp_turns_total 0 → 3`, `vp_chat_seconds_count 3` (flat before this change) |

### Raw run

```text
========== SECTION 1 — HTTP metrics + Redis session store ==========
vp_turns_total (HTTP path): 0.0 -> 3.0   (was flat before this change)
chat-latency samples recorded: 3.0
session key(s) in Redis:
vp:sess:alice
alice conversation history (Redis LIST, role|text):
{"role": "user", "content": "who are you", "at": "2026-07-01T01:34:46.409461+00:00"}
{"role": "assistant", "content": "I’m Ani. My persona background is: ...", ...}
... (6 entries: 3 user + 3 assistant)
alice session TTL: 604800 s  (7-day bound -> idle keys evaporate)

========== SECTION 2 — TTS response cache: miss (full synth) vs hit (Redis GET) ==========
  miss=1.177515s  hit=0.002613s
  miss=0.959132s  hit=0.002508s
  miss=0.974362s  hit=0.002832s
  miss=1.165028s  hit=0.002547s
  --> avg MISS=1069.0 ms | avg HIT=2.62 ms | 407x faster | 99.75% latency cut
  cache counter delta this run: miss +4, hit +4

========== SECTION 3 — per-user token-bucket rate limiting (capacity=5) ==========
burst of 10 rapid requests (one user) -> allowed=5, rejected=5
vp_rate_limited_total: 0.0 -> 5.0
bucket key: vp:rl:burst  TTL=60 s
```

### Redis keys created (DB 15)

```
vp:sess:alice          # session history (RPUSH + LTRIM + 7-day TTL)
vp:sess:burst
vp:rl:alice            # token-bucket state (atomic Lua, 60 s TTL)
vp:rl:burst
vp:tts:3c65ac36...     # cached waveform, key = sha256(provider|text), 24 h TTL
vp:tts:4010cf7c...
vp:tts:547322ac...
vp:tts:949768b9...
```

## Honest notes

- **The cache "miss" is a real synthesis** through the fallback TTS provider
  (~1 s); the "hit" is a single Redis `GET` (~2.6 ms). Same numbers hold for any
  provider — the cache sits in front of CosyVoice/FishAudio identically.
- **`vp_tts_cache_total{result="miss"}` starts non-zero** because the app
  pre-warms filler audio ("嗯…", "让我想想…") into the cache at startup; the demo
  reports the per-run *delta* (miss +4 / hit +4) to isolate its own traffic.
- **Voice-path histograms** (`vp_vad_seconds`, `vp_stt_seconds`,
  `vp_ttfa_seconds`, `vp_tts_seconds`) read 0 here because this demo drives the
  **HTTP** path; they populate on the WebSocket voice path (`app/ws/handler.py`),
  which emits VAD/STT/TTFA/TTS latency and the turn counter.
- **DB 15** is used throughout so the demo is self-contained and disposable; the
  app's live data (db0) is never read or written.
