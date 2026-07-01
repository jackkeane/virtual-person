# Virtual Person (Ani) — Phase 1

A FastAPI-based AI companion app with chat, memory, persona management, and a browser client (Live2D-ready).

## Highlights

- 🧠 Memory system with curation + ranking
- 💾 Durable memory persistence (survives app/vLLM restarts)
- 🗑️ Memory erase flow (backend + UI)
- 🎙️ STT language switch in UI (`zh-CN` / `English` / `Auto`)
- 🧍 Persona profile management (`persona.json`)
- 🔊 Voice pipeline hooks (STT/TTS/VAD + viseme support)
- 🛡️ Safety gate + audit logging
- ⚡ Redis + observability: Prometheus `/metrics`, per-caller rate limiting, TTS response cache (all gated on `REDIS_URL`)
- ✅ CI (GitHub Actions) + offline evaluation harness — safety-gate & retrieval scorecard, gating every push

## Repository Layout

```text
app/         FastAPI backend (chat, memory, tools, voice, ws)
client/      Web client + Live2D assets
tests/       Unit/integration tests
start.sh     Start app (and optionally vLLM)
stop.sh      Stop app + related workers
status.sh    Runtime status helper
```

## Quick Start

### 1) Install

```bash
cd virtual-person-phase1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure (minimal)

```bash
export LLM_PROVIDER='openai_compat'
export OPENAI_COMPAT_BASE_URL='http://127.0.0.1:8000/v1'
export OPENAI_COMPAT_MODEL='Qwen/Qwen3-14B-AWQ'
export OPENAI_COMPAT_API_KEY='dummy'
```

Optional memory DB backends:

```bash
export MEMORY_POSTGRES_DSN='postgresql://user:pass@localhost:5432/virtual_person'
export MEMORY_NEO4J_URI='bolt://localhost:7687'
export MEMORY_NEO4J_USERNAME='neo4j'
export MEMORY_NEO4J_PASSWORD='your-password'
```

### 3) Run

```bash
./start.sh --dev
```

Then open:

- App health: <http://127.0.0.1:8080/health>
- Web client: <http://127.0.0.1:8080/client/>

## Useful Commands

Start with auto vLLM startup:

```bash
./start.sh --dev --with-vllm
```

Low-VRAM voice profile:

```bash
./start.sh --dev --voice-profile
```

Stop services:

```bash
./stop.sh
```

Status:

```bash
./status.sh
```

Terminal chat helper:

```bash
~/anaconda3/bin/conda run -n py312 python chat.py
```

## Recent Changes

### Memory Improvements

- Durable local persistence (`memory_store.json` fallback path)
- Write filtering (noise/duplicate reduction)
- Metadata support in memory model/write API
- Relevance ranking improvements (match + recency + weighting)

### Memory Erase

- API endpoint to erase memory with explicit confirmation
- UI button for erase flow in settings

### STT Language Mode Switch

- UI switch: `zh-CN` / `English` / `Auto`
- Selection persisted in `localStorage`
- Language hint forwarded through WS pipeline to STT

### Restart-Safe Identity Memory

Identity and user facts remain available after app/vLLM restarts via durable memory persistence (validated in tests).

### Redis + Observability (Feature 3)

Backend infrastructure on the realtime voice pipeline, each **gated on `REDIS_URL`** — unset ⇒ in-memory sessions, no cache, allow-all rate limit, byte-identical to before (full suite still `141 passed`):

- **TTS response cache** — Redis-cached waveforms; a hit collapses synth to a ~3 ms Redis `GET` (**≥99.7% latency cut**; a sample run: 2231 ms → 2.9 ms, 763×)
- **Per-caller rate limiting** — atomic Lua token bucket keyed on the **client IP** (not the spoofable `user_id`), fail-open; a 10-request burst yields **5 allowed / 5 rejected**
- **Redis session store** — durable conversation history in `vp:sess:*`, **7-day TTL**
- **Prometheus metrics** — `/metrics` over the HTTP + WS turn paths (`vp_turns_total`, `vp_chat_seconds`, cache / rate-limit counters); optional bearer auth via `METRICS_AUTH_TOKEN`

Reproduce (Redis required; uses DB 15, never db0):

```bash
scripts/demo/serve.sh      # app on :8090 with Redis + ollama + fallback TTS
scripts/demo/run_demo.sh   # drives traffic, prints the numbers
```

Full writeup + captured output: [docs/DEMO.md](./docs/DEMO.md).

## API Notes

Common endpoints:

- `POST /chat/turn`
- `GET /persona/profile`
- `PATCH /persona/profile`
- `GET /memory/search?query=...`
- `POST /memory/write`
- `DELETE /memory/erase?confirm=true`
- `GET /health`
- `GET /metrics` (Prometheus exposition)

## Testing

```bash
~/anaconda3/bin/conda run -n py312 python -m pytest -q
```

**CI** runs the suite on every push/PR (`.github/workflows/ci.yml`): a default job (Redis dormant), a Redis-service job that exercises the Redis-gated paths, and a deterministic **eval** job that publishes a scorecard artifact.

**Evaluation harness** ([`eval/`](./eval/README.md)) — an offline, no-LLM scorecard over the safety gate and memory retrieval, plus an optional local LLM-as-judge quality eval:

```bash
python eval/run_eval.py     # safety-gate + retrieval scorecard -> eval/report.json
```

## Troubleshooting

- If app doesn’t stop cleanly (especially with `--reload`), run `./stop.sh` again.
- If vLLM is already running manually, `start.sh` should detect it.
- If memory seems reset, check configured persistence path and write permissions.

## Future Upgrade Plan

Planned next-step upgrades:

- **Original avatar pipeline**
  - Use assets from `virtual-person-avatar-assets` to build a custom/original avatar.
  - Replace the current default avatar after rigging, expression mapping, and QA.
  - Keep compatibility with existing viseme + emotion blend pipeline.

- **Fine-tuned LLM model**
  - Move from general-purpose base model to a fine-tuned companion model.
  - Target better persona consistency, long-context recall quality, and bilingual (zh/en) dialogue style.
  - Keep fallback to baseline model for stability and A/B comparison.

- **Memory system evolution**
  - Add per-memory item deletion/edit UI.
  - Add memory export/import and backup tooling.
  - Add retrieval quality telemetry and rejection-reason metrics.

- **Voice + latency improvements**
  - Further optimize STT/TTS end-to-end latency.
  - Improve streaming turn-taking and filler strategy.
  - Add more robust handling for low-volume and mixed-language speech.

## License

This project is licensed under the [MIT License](./LICENSE).
