# Virtual Person Phase 1 (MVP Companion)

FastAPI scaffold for Phase 1 of `virtual-person-ani-style-plan.md`.

## Features (Phase 1 + Phase 2)
- Chat turn endpoint using local LLM via Ollama (default) or OpenAI-compatible server
- Memory write/retrieval with **PostgreSQL + Neo4j support** (in-memory fallback)
- **Phase 2: Memory curation** — importance scoring, staleness TTL, dedup, ranked retrieval
- Editable persona background (name / occupation / age / backstory), persisted to `persona.json`
- **Phase 2: Action tools** with risk-tiered permission model (read/write/external + confirmation flow)
- **Phase 2: Reminder CRUD** — add, list, cancel, auto-fire on check
- **Phase 2: Daily summary** endpoint with upcoming/overdue reminders
- Proactivity checks (quiet hours + cooldown + due reminders)
- Safety gate with refusal path
- Audit logging endpoint
- 16 tests covering Phase 1 + Phase 2 features

## Run
```bash
cd ~/clawd/virtual-person-phase1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional DB backends for memory
export MEMORY_POSTGRES_DSN='postgresql://user:pass@localhost:5432/virtual_person'
export MEMORY_NEO4J_URI='bolt://localhost:7687'
export MEMORY_NEO4J_USERNAME='neo4j'
export MEMORY_NEO4J_PASSWORD='your-password'

# LLM runtime provider (default: openai_compat)
export LLM_PROVIDER='openai_compat'
export LLM_TIMEOUT_SECONDS='90'

# OpenAI-compatible local server (vLLM / LM Studio / llama.cpp server)
export OPENAI_COMPAT_BASE_URL='http://127.0.0.1:8001/v1'
export OPENAI_COMPAT_MODEL='Qwen2.5-14B-Instruct'
export OPENAI_COMPAT_API_KEY=''

# Optional: Ollama fallback
export OLLAMA_BASE_URL='http://127.0.0.1:11434'
export OLLAMA_MODEL='qwen3:32b'

# Persona background defaults (persisted to PERSONA_PROFILE_PATH)
export PERSONA_NAME='Ani'
export PERSONA_OCCUPATION='waitress'
export PERSONA_AGE='30'
export PERSONA_BACKSTORY='Ani is an AI companion persona with a calm waitress-style background.'
export PERSONA_PROFILE_PATH='./persona.json'

uvicorn app.main:app --reload
```

Check status:
```bash
curl http://127.0.0.1:8000/memory/backends
curl http://127.0.0.1:8000/persona/profile
```

Update persona background later:
```bash
curl -X PATCH http://127.0.0.1:8000/persona/profile \
  -H 'content-type: application/json' \
  -d '{"name":"Ani","occupation":"barista","age":28,"backstory":"Ani worked late cafe shifts and loves helping people plan clearly."}'
```

Ask for persona story in chat:
```bash
curl -X POST http://127.0.0.1:8000/chat/turn \
  -H 'content-type: application/json' \
  -d '{"user_id":"u1","message":"Tell me about yourself and your background"}'
```

## Test
```bash
cd ~/clawd/virtual-person-phase1
source .venv/bin/activate
pytest -q
```

## Terminal chat (easier than curl)
With server running:
```bash
cd ~/clawd/virtual-person-phase1
~/anaconda3/bin/conda run -n py312 python chat.py
```

## One-click startup (recommended)

```bash
cd ~/clawd/virtual-person-phase1
./start.sh --dev
```

With auto-start vLLM when needed:
```bash
./start.sh --dev --with-vllm
```

Low-VRAM voice mode (recommended when adding TTS on same GPU):
```bash
./start.sh --dev --voice-profile
```

Stop services started by script:
```bash
./stop.sh
```

Check runtime status:
```bash
./status.sh
```

Notes:
- App defaults to `http://127.0.0.1:8080`
- Web client at `http://127.0.0.1:8080/client/`
- Logs written to `./logs/app.log` and `./logs/vllm.log`

## Running with vLLM (current setup)

**Terminal 1 — Start vLLM:**
```bash
cd ~
source .venv/bin/activate
vllm serve Qwen/Qwen3-14B-AWQ --port 8000 --enforce-eager --max-model-len 4096
```

**Terminal 2 — Start Ani app (port 8080, avoids vLLM collision):**
```bash
cd ~/clawd/virtual-person-phase1
~/anaconda3/bin/conda run -n py312 uvicorn app.main:app --port 8080 --reload
```

**Terminal 3 — Chat:**
```bash
cd ~/clawd/virtual-person-phase1
~/anaconda3/bin/conda run -n py312 python chat.py
```

Default config already points at vLLM on port 8000. To switch to Ollama:
```bash
export LLM_PROVIDER=ollama
```

**Note:** Qwen3 outputs `<think>...</think>` reasoning blocks. The app strips them automatically (`LLM_STRIP_THINKING=true` by default).

## Manual verification: Phase 3 Step 2 lip-sync

1. Start app + web client and send one chat turn.
2. In DevTools WS frames, confirm server emits both `audio` and `viseme` timeline events.
3. Watch avatar mouth:
   - Live2D path: viseme values bind to common Cubism mouth params (`ParamMouthOpenY`, `ParamMouthForm`, with probing fallback).
   - Placeholder path: mouth still follows `mouth_open` / `mouth_form` updates.
4. Confirm mouth keeps animating while speech audio is playing and smoothly decays back toward neutral after speech ends.

## Phase 3 Step 3: Mouth/Expression Blend + Smoothing

The avatar runtime now uses layered control instead of direct hard-sets:
- **Base expression layer** (`eyes`, `brows`, `head`, baseline `mouth_open`/`mouth_form`)
- **Speech mouth layer** (viseme-driven `mouth_open` + `mouth_form`, dominant while speaking)
- **Optional micro-motion layer** (small breathing/jitter offsets)

Blending behavior:
- While speaking, speech mouth values dominate through a smooth speech-weight ramp.
- When speaking ends, speech weight decays and mouth smoothly returns to expression baseline.
- Expression changes arriving mid-speech update expression targets, then blend in without abrupt jumps.

### Manual tuning guide

Tune in `client/js/app.js` (`AVATAR_BLEND_TUNING`) or at runtime via:

```js
window.__ANI_AVATAR_TUNING__ = {
  smoothing: { expressionLambda: 7, mouthLambda: 20, speechActivityLambda: 14 },
  microMotion: { enabled: true, freqHz: 0.22, mouthOpenAmp: 0.012, headAmp: 0.015 }
};
```

Quick knobs:
- Mouth too **snappy/chattery** → lower `mouthLambda` (e.g. 20 → 14)
- Mouth too **sluggish/late** → raise `mouthLambda` (e.g. 20 → 26)
- Expression changes too abrupt → lower `expressionLambda`
- Expression changes too slow/floaty → raise `expressionLambda`
- Speaking transitions too abrupt at start/end → lower `speechActivityLambda`
- Idle avatar too still/noisy → adjust `microMotion.*` or disable `microMotion.enabled`
