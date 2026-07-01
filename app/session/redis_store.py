"""Redis-backed conversation session store (Feature 3, worker 2).

Drop-in backend for :class:`app.session.service.SessionService`. It is only
constructed when ``config.session_backend`` selects redis AND Redis is
reachable (see ``SessionService._init_redis``), mirroring the
``app/memory/service.py:_init_postgres`` fallback pattern. When Redis is
unset/unreachable, ``SessionService`` keeps using its in-memory history and
this module is never instantiated.

Public method signatures mirror ``SessionService.add/get/clear`` exactly so
the two backends are interchangeable.

Sync redis (``decode=True``) on purpose -- every caller runs in sync /
threadpool context; the async hot path offloads via ``run_in_executor``
(see ``app/infra/redis_client.py``).

Storage layout: one Redis list per user under ``vp:sess:<user_id>``. Each
element is a JSON object ``{"role", "content", "at"}``. ``add`` appends with
``RPUSH`` then ``LTRIM``s to the most recent ``max_turns`` entries, so an
individual list never grows unbounded. Every write also refreshes a bounded
``EXPIRE`` (``config.session_ttl``) so the *number* of keys stays bounded too:
idle users' keys evaporate instead of accumulating one permanent key per
distinct (unauthenticated, attacker-controllable) ``user_id`` -- matching the
``vp:rl:``/``vp:tts:`` seams.

Fail-soft: every Redis operation is wrapped so a post-startup Redis outage
(the cached client raising ``ConnectionError`` mid-request) degrades to a no-op
(``get`` -> ``[]``; ``add``/``clear`` -> silently skipped) rather than raising
on the realtime path, mirroring ``rate_limit.allow`` and ``CachedTTSService``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

from app.config import config
from app.infra.redis_client import get_sync_redis

# Key namespace for session lists. Kept short to minimize per-op bytes.
_KEY_PREFIX = "vp:sess:"

# Strip newlines and ASCII control chars (C0 range + DEL) from user_id so a
# hostile/garbled id cannot inject key structure or smuggle control bytes into
# the keyspace. Printable text (incl. spaces and unicode letters) is preserved.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class RedisSessionStore:
    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns

    @staticmethod
    def key(user_id: str) -> str:
        """Return the Redis list key for ``user_id`` with control chars stripped."""
        return _KEY_PREFIX + _CONTROL_RE.sub("", user_id or "")

    def add(self, user_id: str, role: str, content: str) -> None:
        client = get_sync_redis(decode=True)
        if client is None:
            # Backend vanished after construction; behave as a no-op rather
            # than raise on the realtime path (gating/passthrough principle).
            return
        k = self.key(user_id)
        payload = json.dumps(
            {"role": role, "content": content, "at": datetime.now(UTC).isoformat()}
        )
        try:
            client.rpush(k, payload)
            # Keep only the most recent ``max_turns`` entries.
            client.ltrim(k, -self.max_turns, -1)
            # Refresh a bounded TTL so idle-user keys evaporate (matches the
            # vp:rl:/vp:tts: seams) instead of one permanent key per user_id.
            client.expire(k, config.session_ttl)
        except Exception:
            # Fail soft: a post-startup Redis outage (cached client raising)
            # must not 500 the chat turn -- drop this write like the in-memory
            # path would lose nothing fatal. (Same contract as the sibling seams.)
            return

    def get(self, user_id: str, limit: int | None = None) -> list[dict]:
        client = get_sync_redis(decode=True)
        if client is None:
            return []
        try:
            raw = client.lrange(self.key(user_id), 0, -1)
        except Exception:
            # Fail soft: a Redis outage degrades to "no history" rather than
            # raising an HTTP 500 on the read path (sibling-seam contract).
            return []
        turns: list[dict] = []
        for item in raw:
            try:
                d = json.loads(item)
            except (ValueError, TypeError):
                # Skip any non-JSON element rather than crash the read path.
                continue
            turns.append({"role": d.get("role"), "content": d.get("content")})
        # Mirror SessionService.get: falsy limit (None/0) returns everything.
        if limit:
            turns = turns[-limit:]
        return turns

    def clear(self, user_id: str) -> None:
        client = get_sync_redis(decode=True)
        if client is None:
            return
        try:
            client.delete(self.key(user_id))
        except Exception:
            # Fail soft: a Redis outage must not 500 the clear-history path.
            return
