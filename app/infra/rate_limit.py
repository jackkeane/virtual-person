"""Per-user token-bucket rate limiting backed by an atomic Redis Lua script.

Feature 3, worker 4. Mirrors the fallback pattern used everywhere else in this
project (see ``app/memory/service.py:_init_postgres`` and
``app/session/service.py:_init_redis``): the limiter is only *active* when a
Redis client is actually available. With ``REDIS_URL`` unset OR Redis
unreachable, :meth:`TokenBucketLimiter.allow` ALWAYS returns ``(True, 0.0)`` so
the realtime path behaves exactly as it did before Feature 3 (allow-all).

Design notes
------------
* **Atomic & skew-free.** Read-modify-write of the bucket happens inside a
  single Lua script, so concurrent turns for the same user can't race. The
  script reads the wall clock from the Redis server itself (``TIME``) rather
  than from any app host, so refill math is immune to client clock skew and to
  multiple app processes disagreeing about "now".

* **Sync redis on purpose.** Like every other Redis seam here, this uses the
  cached *synchronous* client from ``app/infra/redis_client.py``. The async hot
  path (``app/ws/handler.py``) calls :meth:`allow` through
  ``loop.run_in_executor`` so the event loop is never blocked; ``/chat/turn`` is
  a sync endpoint already running in FastAPI's threadpool.

* **Fail OPEN.** Rate limiting is a guard rail, not a correctness mechanism. Any
  Redis hiccup (timeout, connection drop, script error) degrades to "allow"
  rather than rejecting a real user's turn.
"""

from __future__ import annotations

import re

from app.infra.redis_client import get_sync_redis

# Key namespace for per-user buckets. Kept short to minimize per-op bytes.
_KEY_PREFIX = "vp:rl:"

# Strip newlines and ASCII control chars (C0 range + DEL) from user_id so a
# hostile/garbled id cannot smuggle control bytes into the keyspace. Printable
# text (incl. spaces and unicode letters) is preserved. Matches the same
# defensive treatment in app/session/redis_store.py.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# Atomic token-bucket refill+consume.
#   KEYS[1]  = bucket key ("vp:rl:<user_id>")
#   ARGV[1]  = capacity (max tokens)
#   ARGV[2]  = refill rate (tokens / second)
#   ARGV[3]  = requested tokens (always 1 here)
#   ARGV[4]  = key TTL in seconds
# Returns { allowed (0|1), retry_after_seconds (string) }.
# retry_after is returned as a STRING because Redis truncates Lua numbers to
# integers when returned directly; we want sub-second precision on the client.
_LUA_TOKEN_BUCKET = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

-- Server-side clock: { seconds, microseconds } -> float seconds.
local t = redis.call('TIME')
local now = tonumber(t[1]) + (tonumber(t[2]) / 1000000.0)

local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local ts = tonumber(bucket[2])

-- First sighting (or expired): start full.
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now
end

-- Refill for the elapsed wall time, capped at capacity.
local elapsed = now - ts
if elapsed < 0 then
  elapsed = 0
end
tokens = tokens + (elapsed * refill)
if tokens > capacity then
  tokens = capacity
end

local allowed = 0
local retry_after = 0.0
if tokens >= requested then
  tokens = tokens - requested
  allowed = 1
else
  if refill > 0 then
    retry_after = (requested - tokens) / refill
  else
    -- No refill configured: bucket never recovers on its own; surface ttl.
    retry_after = ttl
  end
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, ttl)

return { allowed, tostring(retry_after) }
"""


class TokenBucketLimiter:
    """Per-user token bucket. No-op (allow-all) when Redis is unavailable.

    Parameters
    ----------
    capacity:
        Maximum burst size — the number of tokens a freshly-seen user starts
        with and the ceiling the bucket refills toward. Coerced to ``>= 1``.
    refill_per_sec:
        Steady-state allowance: tokens added per second. Coerced to ``>= 0``.
        ``0`` means "burst of ``capacity`` then nothing until the key expires".
    """

    def __init__(self, capacity: int, refill_per_sec: float) -> None:
        self.capacity = max(1, int(capacity))
        self.refill_per_sec = max(0.0, float(refill_per_sec))
        self._ttl = self._compute_ttl()

    def _compute_ttl(self) -> int:
        """Short TTL so idle buckets evaporate (a re-seeded bucket starts full,
        which is the correct state for an idle user anyway)."""
        if self.refill_per_sec > 0:
            # Comfortably longer than a full refill so an active burst never
            # loses its bucket mid-window, but still bounded.
            return max(60, int(self.capacity / self.refill_per_sec * 2) + 1)
        return 3600

    def _key(self, user_id: str) -> str:
        return _KEY_PREFIX + _CONTROL_RE.sub("", user_id or "")

    def allow(self, user_id: str) -> tuple[bool, float]:
        """Try to consume one token for ``user_id``.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after`` is ``0.0``
        whenever the call is allowed. When Redis is unconfigured/unreachable, or
        on ANY Redis error, this fails OPEN and returns ``(True, 0.0)``.
        """
        client = get_sync_redis(decode=True)
        if client is None:
            # Gating principle: inert without Redis (matches in-memory fallback).
            return (True, 0.0)

        try:
            result = client.eval(
                _LUA_TOKEN_BUCKET,
                1,
                self._key(user_id),
                self.capacity,
                self.refill_per_sec,
                1,
                self._ttl,
            )
            allowed_raw, retry_raw = result[0], result[1]
            if isinstance(retry_raw, bytes):
                retry_raw = retry_raw.decode("utf-8", "ignore")
            allowed = bool(int(allowed_raw))
            if allowed:
                return (True, 0.0)
            retry_after = float(retry_raw)
            if retry_after < 0:
                retry_after = 0.0
            return (False, retry_after)
        except Exception:
            # Fail OPEN: a limiter outage must never take down a real turn.
            return (True, 0.0)
