from __future__ import annotations

import pytest

from app.config import config
from app.infra.redis_client import reset_redis_cache
from app.session.redis_store import RedisSessionStore
from app.session.service import SessionService

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


def _flush_db15() -> None:
    """Flush ONLY db15 (never db0). Safe no-op if redis is unreachable."""
    try:
        import redis as _redis

        _redis.Redis.from_url(REDIS_TEST_URL).flushdb()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Pure-logic tests (no redis required, always run)
# --------------------------------------------------------------------------- #

def test_key_namespacing_and_control_char_stripping() -> None:
    store = RedisSessionStore(max_turns=5)
    assert store.key("alice") == "vp:sess:alice"
    # Newlines, carriage returns and tabs are stripped; spaces/letters survive.
    assert store.key("a\nb\r\tc") == "vp:sess:abc"
    assert store.key("with space") == "vp:sess:with space"
    assert store.key("") == "vp:sess:"


def test_default_backend_is_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    # No REDIS_URL -> redis_available() False -> _redis is None, and the store
    # behaves exactly like the pre-Feature-3 in-memory implementation.
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(config, "redis_url", "")
    monkeypatch.setattr(config, "session_backend", "auto")
    reset_redis_cache()

    svc = SessionService(max_turns=3)
    assert svc._redis is None

    svc.add("u1", "user", "hello")
    svc.add("u1", "assistant", "hi there")
    assert svc.get("u1") == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]

    # max_turns trimming retains only the most recent N turns.
    svc.add("u1", "user", "a")
    svc.add("u1", "user", "b")
    got = svc.get("u1")
    assert len(got) == 3
    assert [t["content"] for t in got] == ["hi there", "a", "b"]

    # limit honored (last N); clear() empties.
    assert svc.get("u1", limit=1) == [{"role": "user", "content": "b"}]
    svc.clear("u1")
    assert svc.get("u1") == []

    reset_redis_cache()


def test_memory_backend_never_uses_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even with a real REDIS_URL set, session_backend="memory" forces the
    # in-memory store (the config gate short-circuits before any ping).
    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)
    monkeypatch.setattr(config, "session_backend", "memory")
    reset_redis_cache()
    try:
        svc = SessionService(max_turns=5)
        assert svc._redis is None
    finally:
        reset_redis_cache()


# --------------------------------------------------------------------------- #
# Redis-path tests (skipped if redis is not reachable)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not _redis_reachable(), reason="redis not reachable at 127.0.0.1:6379"
)
def test_redis_backend_roundtrip_trim_and_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)
    monkeypatch.setattr(config, "session_backend", "redis")
    reset_redis_cache()
    _flush_db15()

    user_id = "redis-user-1"
    try:
        svc = SessionService(max_turns=10)
        assert svc._redis is not None
        assert isinstance(svc._redis, RedisSessionStore)

        # Fresh user starts empty.
        assert svc.get(user_id) == []

        # Add 25 turns; only the last max_turns (10) must survive the LTRIM.
        for i in range(25):
            role = "user" if i % 2 == 0 else "assistant"
            svc.add(user_id, role, f"msg-{i}")

        got = svc.get(user_id)
        # get() returns the last max_turns turns, oldest-first.
        assert len(got) == 10
        assert [t["content"] for t in got] == [f"msg-{i}" for i in range(15, 25)]

        # role/content round-trip faithfully (and keys are exactly role+content).
        assert got[0] == {"role": "assistant", "content": "msg-15"}  # i=15 odd
        assert got[-1] == {"role": "user", "content": "msg-24"}      # i=24 even
        for t in got:
            assert set(t.keys()) == {"role", "content"}

        # limit returns the last N of the retained turns.
        assert [t["content"] for t in svc.get(user_id, limit=3)] == [
            "msg-22",
            "msg-23",
            "msg-24",
        ]

        # clear() empties the user's history.
        svc.clear(user_id)
        assert svc.get(user_id) == []
    finally:
        _flush_db15()
        reset_redis_cache()


@pytest.mark.skipif(
    not _redis_reachable(), reason="redis not reachable at 127.0.0.1:6379"
)
def test_redis_backend_isolated_per_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)
    monkeypatch.setattr(config, "session_backend", "auto")  # auto -> redis when up
    reset_redis_cache()
    _flush_db15()

    try:
        svc = SessionService(max_turns=20)
        assert svc._redis is not None

        svc.add("user-a", "user", "a1")
        svc.add("user-b", "user", "b1")

        assert svc.get("user-a") == [{"role": "user", "content": "a1"}]
        assert svc.get("user-b") == [{"role": "user", "content": "b1"}]

        svc.clear("user-a")
        assert svc.get("user-a") == []
        # Clearing one user must not touch another.
        assert svc.get("user-b") == [{"role": "user", "content": "b1"}]
    finally:
        _flush_db15()
        reset_redis_cache()


# --------------------------------------------------------------------------- #
# Fail-soft: a post-startup Redis outage must never raise on the chat path.   #
# (No real Redis required -- mirrors test_rate_limit.test_fails_open_on_redis_error.)
# --------------------------------------------------------------------------- #

def test_redis_ops_fail_soft_on_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the cached client raises mid-request (Redis restarted / connection
    dropped after the initial ping), get/add/clear must degrade -- get -> [],
    add/clear -> silent no-op -- instead of letting a ConnectionError escape and
    500 the chat turn. Matches the rate_limit / tts_cache fail-soft contract."""

    class _BoomClient:
        def rpush(self, *a, **k):
            raise RuntimeError("connection dropped")

        def ltrim(self, *a, **k):
            raise RuntimeError("connection dropped")

        def expire(self, *a, **k):
            raise RuntimeError("connection dropped")

        def lrange(self, *a, **k):
            raise RuntimeError("connection dropped")

        def delete(self, *a, **k):
            raise RuntimeError("connection dropped")

    monkeypatch.setattr(
        "app.session.redis_store.get_sync_redis", lambda decode=True: _BoomClient()
    )
    store = RedisSessionStore(max_turns=5)
    # None of these may raise; get degrades to an empty history.
    store.add("u", "user", "hello")
    store.clear("u")
    assert store.get("u") == []


@pytest.mark.skipif(
    not _redis_reachable(), reason="redis not reachable at 127.0.0.1:6379"
)
def test_redis_add_sets_bounded_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every write refreshes a bounded EXPIRE so distinct user_ids cannot
    accumulate one permanent key each (CWE-770). The key must carry a positive
    TTL <= config.session_ttl rather than -1 (no expiry)."""
    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)
    monkeypatch.setattr(config, "session_backend", "redis")
    monkeypatch.setattr(config, "session_ttl", 1234)
    reset_redis_cache()
    _flush_db15()
    try:
        store = RedisSessionStore(max_turns=10)
        store.add("ttl-user", "user", "hi")

        import redis as _redis

        raw = _redis.Redis.from_url(REDIS_TEST_URL)
        ttl = raw.ttl("vp:sess:ttl-user")
        assert 0 < ttl <= 1234, f"expected bounded TTL, got {ttl}"
    finally:
        _flush_db15()
        reset_redis_cache()
