"""Unit tests for the per-user token-bucket rate limiter (Feature 3, worker 4).

Coverage:
* No Redis configured  -> ``allow`` ALWAYS returns ``(True, 0.0)`` (the
  pre-Feature-3 allow-all behavior; this is why the default suite, which sets no
  REDIS_URL, stays green).
* Any Redis error       -> fail OPEN (allow), never reject a real turn.
* Real Redis (db 15)    -> capacity is enforced atomically: the first ``capacity``
  calls pass, the next is denied with a positive ``retry_after``.
* Real Redis (db 15)    -> buckets are independent per user_id.

Redis-backed tests point ``config.redis_url`` at db 15, ``pytest.skip`` when Redis
is unreachable, and flush ONLY db 15 on teardown. db 0 is never touched.
"""

from __future__ import annotations

import pytest

from app.config import config
from app.infra.rate_limit import TokenBucketLimiter
from app.infra.redis_client import reset_redis_cache

# Dedicated throwaway DB so tests never collide with real data (which lives on db 0).
REDIS_TEST_URL = "redis://127.0.0.1:6379/15"


def _redis_reachable() -> bool:
    """True iff a real Redis answers PING at the db-15 test URL."""
    try:
        import redis as _redis

        client = _redis.Redis.from_url(
            REDIS_TEST_URL, socket_connect_timeout=0.5, socket_timeout=0.5
        )
        return bool(client.ping())
    except Exception:
        return False


def _flush_db15() -> None:
    """Flush ONLY db 15 (FLUSHDB on a db-15 connection — never FLUSHALL, never db 0)."""
    try:
        import redis as _redis

        client = _redis.Redis.from_url(
            REDIS_TEST_URL, socket_connect_timeout=0.5, socket_timeout=0.5
        )
        client.flushdb()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# No-Redis path: the limiter must be completely inert (allow-all).            #
# --------------------------------------------------------------------------- #

def test_allow_always_true_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """With REDIS_URL unset, allow() returns (True, 0.0) regardless of capacity.

    capacity=1 / refill=0 is deliberately tiny: a Redis-backed bucket would deny
    the 2nd call. Because no client is available the limiter short-circuits to
    allow-all, proving the gating fallback.
    """
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(config, "redis_url", "")
    reset_redis_cache()
    try:
        limiter = TokenBucketLimiter(capacity=1, refill_per_sec=0.0)
        for _ in range(10):
            allowed, retry_after = limiter.allow("anyone")
            assert allowed is True
            assert retry_after == 0.0
    finally:
        # Don't leak the cleared cache to later tests; rebuild lazily next call.
        reset_redis_cache()


def test_fails_open_on_redis_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misbehaving Redis client (eval raises) must degrade to allow, not reject.

    Patches the module-level ``get_sync_redis`` to hand back a client whose
    ``eval`` explodes; the limiter must swallow it and return (True, 0.0). No real
    Redis required, so this runs everywhere.
    """

    class _BoomClient:
        def eval(self, *args, **kwargs):
            raise RuntimeError("redis exploded mid-eval")

    monkeypatch.setattr(
        "app.infra.rate_limit.get_sync_redis", lambda decode=True: _BoomClient()
    )

    limiter = TokenBucketLimiter(capacity=5, refill_per_sec=1.0)
    allowed, retry_after = limiter.allow("u")
    assert allowed is True
    assert retry_after == 0.0


# --------------------------------------------------------------------------- #
# Real-Redis path (db 15). Skipped when Redis is unreachable.                 #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not _redis_reachable(), reason="redis not reachable at 127.0.0.1:6379"
)
def test_token_bucket_denies_after_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """capacity=3: the first 3 calls pass, the 4th is denied with retry_after>0."""
    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)
    reset_redis_cache()
    _flush_db15()  # start from a clean bucket
    try:
        # refill_per_sec=0.5 (one token every 2s) keeps the test deterministic:
        # the 4 back-to-back calls finish in well under a refill interval, so no
        # token is replenished mid-test.
        limiter = TokenBucketLimiter(capacity=3, refill_per_sec=0.5)
        user = "rl_capacity_user"

        for i in range(3):
            allowed, retry_after = limiter.allow(user)
            assert allowed is True, f"call {i} within capacity should be allowed"
            assert retry_after == 0.0

        # Bucket now drained -> deny, and tell the caller (roughly) how long to wait.
        allowed, retry_after = limiter.allow(user)
        assert allowed is False
        assert retry_after > 0.0
    finally:
        _flush_db15()
        reset_redis_cache()


@pytest.mark.skipif(
    not _redis_reachable(), reason="redis not reachable at 127.0.0.1:6379"
)
def test_buckets_are_per_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """Draining one user's bucket must not affect another user's bucket."""
    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)
    reset_redis_cache()
    _flush_db15()
    try:
        limiter = TokenBucketLimiter(capacity=2, refill_per_sec=0.5)

        # Drain user A completely.
        assert limiter.allow("user_a")[0] is True
        assert limiter.allow("user_a")[0] is True
        assert limiter.allow("user_a")[0] is False  # A is now rate-limited

        # User B has an independent, still-full bucket.
        assert limiter.allow("user_b")[0] is True
        assert limiter.allow("user_b")[0] is True
    finally:
        _flush_db15()
        reset_redis_cache()
