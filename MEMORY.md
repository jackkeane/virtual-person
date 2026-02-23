# Memory System (Current State)

This document describes the memory stack in **Virtual Person (Ani) Phase 1** after Day 6 implementation and follow-up fixes.

## Goals

- Keep important user facts across restarts
- Minimize noisy/incorrect memory writes
- Improve retrieval relevance and determinism
- Support explicit user memory erase
- Keep DB and fallback persistence both reliable

## Active Architecture

Memory is managed by `app/memory/service.py` with layered storage:

1. **PostgreSQL** (when `MEMORY_POSTGRES_DSN` is configured and reachable)
2. **Neo4j** (when `MEMORY_NEO4J_*` env vars are configured and reachable)
3. **File-backed fallback** at a stable path:
   - `virtual-person-phase1/memory_store.json`

The fallback remains available even if DB backends are down.

## Day 6 + Follow-up Improvements

### A) Durable persistence

- Stable absolute default path for fallback memory file
- Atomic flush strategy prevents partial writes

### B) Memory curation

- Noise filtering for low-signal writes
- Duplicate suppression by normalized comparison
- Metadata passthrough for provenance/context

### C) Retrieval improvements

- Better ranking (text relevance + recency + weighting)
- User-scoped name recall logic to avoid cross-user contamination

### D) Name memory reliability

- Auto-capture names from natural chat forms:
  - `my name is ...`
  - `I am ...` / `I'm ...`
  - `call me ...`
  - `我是 ...` / `我叫 ...`
- Invalid extraction guard blocks noisy pseudo-names such as:
  - `什么`, `什么名字`, etc.

### E) User controls

- Erase all memory endpoint with confirmation
- UI button + confirmation in Avatar Settings

### F) Runtime debugability

- Added `GET /memory/debug?user_id=...` for quick diagnosis:
  - active persist path
  - file existence
  - stored `name` entries
  - currently recalled name for user

## Operational Behavior

- `start.sh` loads `.env`
- `.env_example` provided for reproducible setup
- With correct DSN/credentials, `/memory/backends` should report:
  - `postgres: true`
  - `neo4j: true`

## Verified Restart Behavior

Live check completed:

- DB backends true before restart
- Test memory written
- `./stop.sh` then `./start.sh --dev`
- DB backends still true after restart
- Test memory still retrievable

This confirms restart persistence works in the current configured setup.

## Key Endpoints

- `POST /memory/write`
- `GET /memory/search?query=...`
- `DELETE /memory/erase?confirm=true`
- `GET /memory/backends`
- `GET /memory/debug?user_id=web_user`
