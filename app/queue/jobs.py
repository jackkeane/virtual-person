"""Background job registry + handlers (Feature: async background jobs).

Each handler runs OFF the realtime voice path, on a standalone consumer worker.
Handlers do REAL work against the same services the HTTP app uses
(``MemoryService``, ``ProactivityService``, ``app.memory.curation``); they are
idempotent where feasible so an at-least-once redelivery is safe.

Failure contract: handlers let genuine failures PROPAGATE — the queue's consumer
turns an exception into a retry and, past ``queue_max_retries``, a dead-letter
(see ``app/queue/task_queue.py``). The sub-services these handlers call are
themselves fail-soft (``memory.delete``/``write`` return bools, never raise), so
in practice only :func:`poison` raises — it exists solely to exercise the DLQ.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Callable

from app.config import config
from app.memory.curation import ScoredMemory, is_stale, rank_memories, retrieval_score
from app.memory.service import MemoryService
from app.proactivity.service import ProactivityService
from app.queue.task_queue import enqueue as _enqueue

logger = logging.getLogger(__name__)

# --- Task type names (the wire ``type`` field; shared with producers) --------- #
CURATE_MEMORIES = "curate_memories"
DAILY_SUMMARY_PRECOMPUTE = "daily_summary_precompute"
PROACTIVE_NUDGE = "proactive_nudge"
POISON = "poison"


# --------------------------------------------------------------------------- #
# Service accessors
# --------------------------------------------------------------------------- #
# Prefer the LIVE singletons from app.main when the app is already imported in
# this process (the same-process demo: the consumer sees reminders added via the
# HTTP API and shares the in-memory ProactivityService). We only READ sys.modules
# — we never import app.main ourselves, so a standalone worker stays lightweight
# and free of the FastAPI app's import side effects. In a separate worker process
# these fall back to fresh instances; MemoryService reloads from the same on-disk
# store, while ProactivityService state is necessarily per-process (in-memory).
_memory: MemoryService | None = None
_proactivity: ProactivityService | None = None


def _get_memory() -> MemoryService:
    main = sys.modules.get("app.main")
    live = getattr(main, "memory", None) if main is not None else None
    if isinstance(live, MemoryService):
        return live
    global _memory
    if _memory is None:
        _memory = MemoryService()
    return _memory


def _get_proactivity() -> ProactivityService:
    main = sys.modules.get("app.main")
    live = getattr(main, "proactivity", None) if main is not None else None
    if isinstance(live, ProactivityService):
        return live
    global _proactivity
    if _proactivity is None:
        _proactivity = ProactivityService(
            quiet_start=config.quiet_hours_start,
            quiet_end=config.quiet_hours_end,
            cooldown_minutes=config.proactive_cooldown_minutes,
        )
    return _proactivity


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def curate_memories(payload: dict) -> dict:
    """Re-rank the memory store and prune stale items.

    Idempotent: ranking is read-only and pruning uses ``memory.delete`` (a
    no-op once an item is already gone), so a redelivery converges to the same
    state. ``payload`` may carry an optional ``query`` to bias the ranking.
    """
    memory = _get_memory()
    query = (payload or {}).get("query", "") or ""

    items = memory.search("")  # "" enumerates ALL items (see main.py:654)
    scored = [
        ScoredMemory(
            kind=m.kind,
            key=m.key,
            value=m.value,
            created_at=m.created_at,
            score=retrieval_score(m.kind, m.key, m.value, m.created_at, query),
            source=getattr(m, "source", "in-memory"),
        )
        for m in items
    ]
    ranked = rank_memories(scored, limit=8)

    pruned = 0
    for m in items:
        if is_stale(m.kind, m.created_at) and memory.delete(m.kind, m.key):
            pruned += 1

    logger.info("curate_memories: scanned=%d ranked=%d pruned=%d", len(items), len(ranked), pruned)
    return {"scanned": len(items), "ranked": len(ranked), "pruned": pruned}


def daily_summary_precompute(payload: dict) -> dict:
    """Precompute the daily summary and persist it for cross-process retrieval.

    Mirrors the ``/daily-summary`` endpoint (main.py:652-657), then caches the
    result as a dated ``note`` memory so a separate worker's output is durable
    (ProactivityService state is per-process). Idempotent per day via a
    delete-then-write upsert on the dated key: at most one (latest) cached
    summary per day, so redelivery never accumulates notes.
    """
    memory = _get_memory()
    proactivity = _get_proactivity()

    recent = memory.search("")
    mem_items = [{"kind": m.kind, "key": m.key, "value": m.value} for m in recent[:20]]
    summary = proactivity.daily_summary(memory_items=mem_items)

    try:
        day = datetime.now(UTC).date().isoformat()
        key = f"daily_summary:{day}"
        memory.delete("note", key)  # upsert: keep a single latest summary per day
        memory.write(
            "note",
            key,
            json.dumps(summary, ensure_ascii=False),
            metadata={"generated_by": DAILY_SUMMARY_PRECOMPUTE},
        )
    except Exception:
        # Persistence is best-effort; the computed summary is still returned.
        pass

    logger.info(
        "daily_summary_precompute: reminders=%d memories=%d",
        summary.get("active_reminder_count", 0),
        len(mem_items),
    )
    return summary


def proactive_nudge(payload: dict) -> dict:
    """Fire due reminders, honoring the anti-annoyance gate.

    Mirrors main.py's /proactivity/check + /proactivity/mark-sent EXACTLY:
    ``can_send``/``mark_sent`` take a NAIVE ``datetime.now()`` (how the endpoint
    stamps ``_last_sent_at``), while ``check_due()`` is called with NO argument so
    it uses its aware-UTC default matching the aware ``due_at`` set by
    ``add_reminder`` — avoiding any naive/aware comparison error on the shared
    singleton. Idempotent: ``check_due`` marks reminders fired, so a redelivery
    does not re-emit them.
    """
    proactivity = _get_proactivity()

    now = datetime.now()  # naive, matches ProactivityService.mark_sent call sites
    ok, reason = proactivity.can_send(now)
    if not ok:
        logger.info("proactive_nudge: suppressed (%s)", reason)
        return {"sent": False, "reason": reason, "due": []}

    due = proactivity.check_due()  # no arg -> aware-UTC default (matches due_at)
    if not due:
        return {"sent": False, "reason": "no-due-reminders", "due": []}

    # "Emit" step: in a background worker there is no live socket, so surfacing the
    # due reminders (returned + logged) is the delivery. Then record the send so
    # the cooldown gate applies to the next nudge.
    proactivity.mark_sent(now)
    emitted = [{"id": r.id, "message": r.message, "due_at": r.due_at.isoformat()} for r in due]
    logger.info("proactive_nudge: emitted %d reminder(s)", len(emitted))
    return {"sent": True, "reason": "ok", "due": emitted}


def poison(payload: dict) -> None:
    """Always raise — exercises the retry + dead-letter (DLQ) path for demos."""
    raise RuntimeError("poison task: intentional failure to exercise the retry/DLQ path")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, Callable[[dict], object]] = {
    CURATE_MEMORIES: curate_memories,
    DAILY_SUMMARY_PRECOMPUTE: daily_summary_precompute,
    PROACTIVE_NUDGE: proactive_nudge,
    POISON: poison,
}


def get_registry() -> dict[str, Callable[[dict], object]]:
    """Return a copy of the ``{type: handler}`` registry (used by the consumer)."""
    return dict(_REGISTRY)


def get_handler(task_type: str) -> Callable[[dict], object] | None:
    """Look up a single handler by task type, or ``None`` if unregistered."""
    return _REGISTRY.get(task_type)


# --------------------------------------------------------------------------- #
# Typed producer helpers (fire-and-forget; never raise; return published?)
# --------------------------------------------------------------------------- #
def enqueue_curate_memories(user_id: str = "") -> bool:
    """Enqueue a memory-curation job (safe to call from the HTTP boundary)."""
    return _enqueue(CURATE_MEMORIES, {"user_id": user_id})


def enqueue_daily_summary_precompute() -> bool:
    """Enqueue a daily-summary precompute job."""
    return _enqueue(DAILY_SUMMARY_PRECOMPUTE, {})


def enqueue_proactive_nudge() -> bool:
    """Enqueue a proactive-nudge job."""
    return _enqueue(PROACTIVE_NUDGE, {})


def enqueue_poison() -> bool:
    """Enqueue a poison job (for the DLQ demonstration)."""
    return _enqueue(POISON, {})
