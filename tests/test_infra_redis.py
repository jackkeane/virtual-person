from __future__ import annotations

import pytest

from app.config import config
from app.infra.redis_client import get_sync_redis, redis_available, reset_redis_cache

REDIS_TEST_URL = "redis://127.0.0.1:6379/15"


def _redis_reachable() -> bool:
    try:
        import redis as _redis

        client = _redis.Redis.from_url(
            REDIS_TEST_URL, socket_connect_timeout=0.5, socket_timeout=0.5
        )
        return bool(client.ping())
    except Exception:
        return False


def test_returns_none_when_redis_url_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mirror the postgres fallback: empty URL -> None, no network attempt.
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(config, "redis_url", "")
    reset_redis_cache()

    assert get_sync_redis() is None
    assert get_sync_redis(decode=False) is None
    assert redis_available() is False

    reset_redis_cache()


@pytest.mark.skipif(not _redis_reachable(), reason="redis not reachable at 127.0.0.1:6379")
def test_returns_working_client_with_db15(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)
    reset_redis_cache()

    key = "ani:test:infra:k"
    try:
        client = get_sync_redis()
        assert client is not None
        assert client.ping() is True
        assert redis_available() is True

        # Decoding client round-trips str.
        client.set(key, "v")
        assert client.get(key) == "v"

        # Binary client is a distinct, separately-cached instance returning bytes.
        bin_client = get_sync_redis(decode=False)
        assert bin_client is not None
        assert bin_client is not client
        assert bin_client.get(key) == b"v"

        # Same flag returns the cached instance (built at most once).
        assert get_sync_redis() is client
        assert get_sync_redis(decode=False) is bin_client
    finally:
        # Surgical cleanup: only the keys this test wrote, only on db15.
        try:
            c = get_sync_redis()
            if c is not None:
                c.delete(key)
        except Exception:
            pass
        reset_redis_cache()


def test_reset_clears_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no URL the cached value is None; after pointing at a URL and resetting,
    # the next call must re-evaluate rather than serve the stale None.
    monkeypatch.setattr(config, "redis_url", "")
    reset_redis_cache()
    assert get_sync_redis() is None

    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)
    # Without reset the stale None is still cached.
    assert get_sync_redis() is None

    reset_redis_cache()
    if _redis_reachable():
        assert get_sync_redis() is not None
    reset_redis_cache()
