"""Redis-backed TTS response cache (Feature 3, worker 5).

Wraps any :class:`BaseTTSService` so that identical ``(provider, text)`` requests
reuse a previously-synthesized waveform instead of re-running the model. Mirrors
the fallback pattern used everywhere else in this project
(``app/memory/service.py:_init_postgres``, ``app/infra/rate_limit.py``): the
cache is only *active* when a Redis client is actually available. With
``REDIS_URL`` unset OR Redis unreachable OR ``config.tts_cache_enabled`` false,
:meth:`CachedTTSService.synthesize` is a transparent passthrough to the wrapped
service and behaves exactly as it did before Feature 3.

Design notes
------------
* **Sync redis on purpose.** Like every Redis seam here, this uses the cached
  *binary* (non-decoding) synchronous client from ``app/infra/redis_client.py``.
  The async hot path (``app/ws/handler.py``) calls ``synthesize`` through
  ``loop.run_in_executor``, so the single cache ``GET`` runs off the event loop:
  on a hit it SAVES the full synthesis time, on a miss it adds only one cheap
  round-trip.

* **Never break TTS.** Audio is correctness-critical; the cache is not. Any Redis
  hiccup (timeout, connection drop, corrupt entry) degrades to a plain
  ``inner.synthesize`` rather than failing the turn. A read miss/error -> synth +
  best-effort write; a write error is swallowed.

* **Key = sha256(provider | text).** Namespaced under ``vp:tts:``. The *wrapped*
  provider name is part of the digest so two providers never collide on cache
  entries, and the stored JSON carries base64 audio + phoneme timestamps so the
  exact ``(bytes, list[dict])`` tuple round-trips.
"""

from __future__ import annotations

import base64
import hashlib
import json

from app.config import config
from app.infra.redis_client import get_sync_redis
from app.observability.metrics import inc_tts_cache
from app.voice.tts_service import BaseTTSService

# Short namespace prefix to minimize per-key bytes, consistent with the other
# Redis seams (``vp:sess:``, ``vp:rl:``).
_KEY_PREFIX = "vp:tts:"


class CachedTTSService(BaseTTSService):
    """Wrap an inner :class:`BaseTTSService` with a Redis-backed response cache.

    Passthrough (no caching) whenever ``config.tts_cache_enabled`` is false or no
    Redis client is available. ``provider_name`` advertises the wrapped provider
    with a ``+cache`` suffix so logs/metrics make the wrapping visible.
    """

    def __init__(self, inner: BaseTTSService) -> None:
        self.inner = inner
        self.provider_name = f"{inner.provider_name}+cache"

    def _key(self, text: str) -> str:
        """Cache key for ``text`` under the *wrapped* provider's identity."""
        digest = hashlib.sha256(
            (self.inner.provider_name + "|" + text).encode("utf-8")
        ).hexdigest()
        return _KEY_PREFIX + digest

    def synthesize(self, text: str) -> tuple[bytes, list[dict]]:
        # Gate: inert without the feature flag or without a live Redis client,
        # so a default ``pytest`` run (no REDIS_URL) behaves exactly as before.
        if not config.tts_cache_enabled:
            return self.inner.synthesize(text)
        client = get_sync_redis(decode=False)
        if client is None:
            return self.inner.synthesize(text)

        key = self._key(text)

        # --- Cache GET ---------------------------------------------------- #
        try:
            cached = client.get(key)
        except Exception:
            # Redis read failed: never break TTS, just synthesize uncached.
            return self.inner.synthesize(text)

        if cached is not None:
            try:
                payload = json.loads(cached)
                audio = base64.b64decode(payload["audio_b64"])
                phonemes = payload.get("phonemes") or []
                inc_tts_cache(hit=True)
                return audio, phonemes
            except Exception:
                # Corrupt/unreadable entry: fall through to a fresh synthesis,
                # which self-heals the key via the SETEX below.
                pass

        # --- Cache MISS: synthesize, best-effort write, count, return ----- #
        audio, phonemes = self.inner.synthesize(text)
        try:
            payload = json.dumps(
                {
                    "audio_b64": base64.b64encode(audio).decode("ascii"),
                    "phonemes": phonemes,
                }
            )
            client.setex(key, config.tts_cache_ttl, payload)
        except Exception:
            # A write failure must not affect the returned audio.
            pass
        inc_tts_cache(hit=False)
        return audio, phonemes
