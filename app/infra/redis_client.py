"""Lazy, process-cached synchronous Redis client factory.

Mirrors the fallback pattern in app/memory/service.py:_init_postgres — when the
configured DSN/URL is unset OR the backend is unreachable, callers transparently
fall back to in-memory / no-op behavior. Everything here is gated on
``config.redis_url`` being non-empty AND a successful ``ping()``; otherwise we
return ``None`` and cache that ``None`` for the lifetime of the process.

Sync (not async) clients are used on purpose: every seam that touches Redis
(SessionService.add/get/clear, BaseTTSService.synthesize) is called from
sync / threadpool context, and the async hot path offloads blocking work via
``loop.run_in_executor``. Async redis would force rewriting those call sites.
"""

from __future__ import annotations

import redis

from app.config import config

# Cache keyed by the ``decode`` flag: we need a decoding client for text values
# and a separate binary (non-decoding) client for audio bytes. A present key
# (even mapping to ``None``) means "already attempted" so we never re-ping a
# known-bad backend on the hot path.
_clients: dict[bool, "redis.Redis | None"] = {}


def get_sync_redis(decode: bool = True) -> "redis.Redis | None":
    """Return a cached, ping-verified ``redis.Redis`` or ``None``.

    Returns ``None`` (and caches it) when ``config.redis_url`` is empty or when
    building / pinging the client raises for any reason. Cached per ``decode``
    flag so text and binary clients are built at most once each per process.
    """
    if decode in _clients:
        return _clients[decode]

    url = config.redis_url
    if not url:
        _clients[decode] = None
        return None

    try:
        client = redis.Redis.from_url(
            url,
            decode_responses=decode,
            socket_timeout=0.5,
            socket_connect_timeout=0.5,
        )
        client.ping()
    except Exception:
        # Cache the failure so the realtime path doesn't pay a connect timeout
        # on every call. reset_redis_cache() clears this after config changes.
        _clients[decode] = None
        return None

    _clients[decode] = client
    return client


def reset_redis_cache() -> None:
    """Clear both cached clients. Call after monkeypatching env/config in tests."""
    _clients.clear()


def redis_available() -> bool:
    """True iff a decoding Redis client could be built and pinged."""
    return get_sync_redis(True) is not None
