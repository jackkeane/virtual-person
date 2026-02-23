# Memory System

This document explains how memory works in **Virtual Person (Ani) Phase 1**, including the Day 6 upgrades.

## Goals

- Preserve important user facts across restarts
- Reduce noisy or duplicated memory writes
- Improve retrieval relevance in real conversations
- Give users explicit erase control from UI

## Architecture

Memory is served by `app/memory/service.py` and can use:

1. **PostgreSQL store** (if configured)
2. **Neo4j store** (if configured)
3. **Durable file-backed fallback** (`memory_store.json`) when DBs are unavailable

This ensures memory continuity even when app and vLLM are restarted.

## Day 6 Improvements

### 1) Durable persistence

- Added persistent fallback storage (file-backed)
- Uses atomic writes (`.tmp` + replace) to avoid partial/corrupt writes

### 2) Better write curation

- Drops low-signal/noisy entries
- Prevents near-duplicate writes via normalized matching
- Supports memory metadata for richer context

### 3) Better retrieval relevance

Ranking combines:

- token overlap
- exact key match bonus
- recency bonus
- kind/importance weighting

### 4) Erase support

- Added backend erase-all path with explicit confirmation
- UI includes an erase button + confirm flow

### 5) Restart-safe identity memory

User identity facts are persisted durably and verified by tests to survive process restarts.

## API Notes

Typical endpoints:

- `GET /memory/search?query=...`
- `POST /memory/write`
- `DELETE /memory/erase?confirm=true`

## Operational Notes

- Back up `memory_store.json` if you rely on file-backed persistence
- For production workloads, prefer database backends + periodic backup
- Keep erase actions explicit and intentional

## Test Coverage (relevant)

Memory behavior is validated by tests including restart-safety and service-level checks in:

- `tests/test_memory_service.py`
- `tests/test_core.py`
