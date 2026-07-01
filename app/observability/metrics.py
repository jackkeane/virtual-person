"""Prometheus metrics for the Ani realtime voice pipeline (Feature 3, worker 3).

Design notes
------------
* Collectors live on the **default global** ``REGISTRY`` so ``generate_latest()``
  (and standard Prometheus tooling) picks them up with no extra wiring, and so a
  single ``/metrics`` scrape exposes both these app metrics and the default
  process/GC collectors.

* **Duplicate-registration guard.** A normally-cached module body runs exactly
  once per process, so a plain module-level definition is already safe across the
  many ``from app.observability.metrics import ...`` calls made by handler.py,
  main.py and the tests. To stay robust even under ``importlib.reload`` or any
  accidental re-execution (which is what triggers prometheus_client's
  ``ValueError: Duplicated timeseries in CollectorRegistry``), every collector is
  built via :func:`_get_or_create`, which reuses the already-registered collector
  instead of raising.

* **Never raise into the hot path.** Every public ``observe_*`` / ``inc_*``
  helper is wrapped by :func:`_never_raises`. Instrumentation must never take down
  a turn: a bad value or a registry hiccup degrades to a no-op, not an exception.
  Calls are O(1) and in-process, keeping the realtime path latency-neutral.
"""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)

# Latency buckets spanning ~10ms .. 10s. This brackets the pipeline's measured
# range: VAD/STT land in the tens-to-hundreds of ms, while TTFA/TTS run from a
# few hundred ms up to a couple of seconds (see LATENCY_BASELINE_*.md).
_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

_C = TypeVar("_C")


def _get_or_create(factory: Callable[..., _C], name: str, *args, **kwargs) -> _C:
    """Build a collector named ``name`` on the default REGISTRY, idempotently.

    On the first call this just constructs the collector. If the same name is
    already registered (module reloaded / re-executed inside a test session),
    prometheus_client raises ``ValueError`` from ``REGISTRY.register``; we catch
    it and return the existing collector. The constructor name is always a key in
    ``REGISTRY._names_to_collectors`` (verified for both Histogram and labeled
    Counter), so the lookup is reliable.
    """
    try:
        return factory(name, *args, **kwargs)
    except ValueError:
        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]


def _never_raises(fn):
    """Wrap an instrumentation helper so it can never raise into the hot path."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    return wrapper


# --- Collectors (defined once; reused on re-import) ---

vp_vad_seconds: Histogram = _get_or_create(
    Histogram, "vp_vad_seconds", "Voice-activity-detection latency in seconds.", buckets=_LATENCY_BUCKETS
)
vp_stt_seconds: Histogram = _get_or_create(
    Histogram, "vp_stt_seconds", "Speech-to-text latency in seconds.", buckets=_LATENCY_BUCKETS
)
vp_ttfa_seconds: Histogram = _get_or_create(
    Histogram, "vp_ttfa_seconds", "Time-to-first-audio per turn in seconds.", buckets=_LATENCY_BUCKETS
)
vp_tts_seconds: Histogram = _get_or_create(
    Histogram, "vp_tts_seconds", "Text-to-speech synthesis latency in seconds.", buckets=_LATENCY_BUCKETS
)
vp_chat_seconds: Histogram = _get_or_create(
    Histogram, "vp_chat_seconds", "End-to-end /chat/turn (text) latency in seconds.", buckets=_LATENCY_BUCKETS
)

vp_turns_total: Counter = _get_or_create(
    Counter, "vp_turns_total", "Completed conversation turns."
)
vp_tts_cache_total: Counter = _get_or_create(
    Counter, "vp_tts_cache_total", "TTS cache lookups, labeled by result.", ["result"]
)
vp_rate_limited_total: Counter = _get_or_create(
    Counter, "vp_rate_limited_total", "Requests rejected by the rate limiter."
)

# Background task-queue lifecycle (Feature: async background jobs). One labeled
# family keeps the exposition compact: `event` is one of enqueued/processed/
# failed/retried/dead_lettered. The producer side (chat_turn HTTP boundary) emits
# `enqueued`; the consumer worker emits the processed/failed/retried/dead_lettered
# transitions. Labeled counters stay dormant (no series) until first incremented,
# exactly like vp_tts_cache_total.
vp_queue_jobs_total: Counter = _get_or_create(
    Counter, "vp_queue_jobs_total", "Background queue job lifecycle events, labeled by event.", ["event"]
)


# --- Instrumentation helpers (never raise) ---

@_never_raises
def observe_vad(seconds: float) -> None:
    """Record a VAD latency sample (seconds)."""
    vp_vad_seconds.observe(seconds)


@_never_raises
def observe_stt(seconds: float) -> None:
    """Record an STT latency sample (seconds)."""
    vp_stt_seconds.observe(seconds)


@_never_raises
def observe_ttfa(seconds: float) -> None:
    """Record a time-to-first-audio sample (seconds)."""
    vp_ttfa_seconds.observe(seconds)


@_never_raises
def observe_tts(seconds: float) -> None:
    """Record a TTS synthesis latency sample (seconds)."""
    vp_tts_seconds.observe(seconds)


@_never_raises
def observe_chat(seconds: float) -> None:
    """Record an end-to-end text chat-turn (/chat/turn) latency sample (seconds)."""
    vp_chat_seconds.observe(seconds)


@_never_raises
def inc_turn() -> None:
    """Count one completed conversation turn."""
    vp_turns_total.inc()


@_never_raises
def inc_tts_cache(hit: bool) -> None:
    """Count one TTS cache lookup as a hit or a miss."""
    vp_tts_cache_total.labels(result="hit" if hit else "miss").inc()


@_never_raises
def inc_rate_limited() -> None:
    """Count one request rejected by the rate limiter."""
    vp_rate_limited_total.inc()


@_never_raises
def inc_job_enqueued() -> None:
    """Count one background job handed to the broker at the HTTP boundary."""
    vp_queue_jobs_total.labels(event="enqueued").inc()


@_never_raises
def inc_job_processed() -> None:
    """Count one background job whose handler completed successfully."""
    vp_queue_jobs_total.labels(event="processed").inc()


@_never_raises
def inc_job_failed() -> None:
    """Count one background job whose handler raised."""
    vp_queue_jobs_total.labels(event="failed").inc()


@_never_raises
def inc_job_retried() -> None:
    """Count one background job requeued for another attempt."""
    vp_queue_jobs_total.labels(event="retried").inc()


@_never_raises
def inc_job_dead_lettered() -> None:
    """Count one background job routed to the dead-letter queue (DLQ)."""
    vp_queue_jobs_total.labels(event="dead_lettered").inc()


def render() -> tuple[bytes, str]:
    """Return ``(exposition_bytes, content_type)`` for the ``/metrics`` endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
