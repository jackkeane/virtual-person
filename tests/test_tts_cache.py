from __future__ import annotations

import pytest

from app.config import config
from app.infra.redis_client import reset_redis_cache
from app.voice.tts_cache import CachedTTSService
from app.voice.tts_service import BaseTTSService

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


class FakeTTS(BaseTTSService):
    """Deterministic inner TTS with a call counter, so cache hits are observable.

    ``synthesize`` increments ``calls`` and returns text-derived bytes plus a
    JSON-safe phoneme list, so a cache hit (which skips this method entirely)
    must leave ``calls`` unchanged while round-tripping identical output.
    """

    provider_name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, text: str) -> tuple[bytes, list[dict]]:
        self.calls += 1
        audio = ("AUDIO:" + text).encode("utf-8")
        phonemes = [{"phoneme": "T", "start_ms": 0, "end_ms": 100}]
        return audio, phonemes


# --------------------------------------------------------------------------- #
# Passthrough tests (no redis required, always run)
# --------------------------------------------------------------------------- #

def test_provider_name_advertises_cache_wrapper() -> None:
    svc = CachedTTSService(FakeTTS())
    assert svc.provider_name == "fake+cache"


def test_passthrough_when_cache_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # tts_cache_enabled False -> inner is called every time; Redis is never touched
    # (the flag short-circuits before any get_sync_redis() call).
    monkeypatch.setattr(config, "tts_cache_enabled", False)
    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)  # set, yet ignored
    reset_redis_cache()
    try:
        inner = FakeTTS()
        svc = CachedTTSService(inner)

        a1, p1 = svc.synthesize("hello")
        a2, p2 = svc.synthesize("hello")

        assert inner.calls == 2
        assert a1 == a2 == b"AUDIO:hello"
        assert p1 == p2
    finally:
        reset_redis_cache()


def test_passthrough_when_no_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    # Enabled but no REDIS_URL -> get_sync_redis() is None -> inner called each call,
    # exactly like the pre-Feature-3 (uncached) behavior.
    monkeypatch.setattr(config, "tts_cache_enabled", True)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(config, "redis_url", "")
    reset_redis_cache()
    try:
        inner = FakeTTS()
        svc = CachedTTSService(inner)

        svc.synthesize("hello")
        svc.synthesize("hello")

        assert inner.calls == 2
    finally:
        reset_redis_cache()


def test_get_tts_service_unwrapped_without_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.voice import tts_service as tts_mod

    monkeypatch.setattr(config, "tts_cache_enabled", True)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(config, "redis_url", "")
    monkeypatch.setattr(config, "tts_provider", "fallback")
    reset_redis_cache()
    try:
        svc = tts_mod.get_tts_service()
        assert not isinstance(svc, CachedTTSService)
    finally:
        reset_redis_cache()


# --------------------------------------------------------------------------- #
# Redis-path tests (skipped if redis is not reachable)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not _redis_reachable(), reason="redis not reachable at 127.0.0.1:6379"
)
def test_cache_miss_then_hit_skips_inner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "tts_cache_enabled", True)
    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)
    reset_redis_cache()
    _flush_db15()
    try:
        inner = FakeTTS()
        svc = CachedTTSService(inner)

        text = "缓存命中测试 cache hit"

        # First call: MISS -> inner runs exactly once and the result is cached.
        a1, p1 = svc.synthesize(text)
        assert inner.calls == 1

        # Second identical call: HIT -> inner NOT called again (counter still 1),
        # and the bytes/phonemes round-trip identically from Redis.
        a2, p2 = svc.synthesize(text)
        assert inner.calls == 1
        assert a2 == a1
        assert p2 == p1

        # A different text is a separate MISS, so inner runs again.
        svc.synthesize("another distinct text")
        assert inner.calls == 2
    finally:
        _flush_db15()
        reset_redis_cache()


@pytest.mark.skipif(
    not _redis_reachable(), reason="redis not reachable at 127.0.0.1:6379"
)
def test_cache_persists_across_wrapper_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The cache lives in Redis, not in the wrapper, so a fresh CachedTTSService
    # over a fresh inner still serves a hit from a prior synthesis.
    monkeypatch.setattr(config, "tts_cache_enabled", True)
    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)
    reset_redis_cache()
    _flush_db15()
    try:
        first_inner = FakeTTS()
        a1, _ = CachedTTSService(first_inner).synthesize("shared")
        assert first_inner.calls == 1

        second_inner = FakeTTS()
        a2, _ = CachedTTSService(second_inner).synthesize("shared")
        # Second wrapper's inner is never invoked: served from Redis.
        assert second_inner.calls == 0
        assert a2 == a1
    finally:
        _flush_db15()
        reset_redis_cache()


@pytest.mark.skipif(
    not _redis_reachable(), reason="redis not reachable at 127.0.0.1:6379"
)
def test_get_tts_service_wraps_when_redis_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.voice import tts_service as tts_mod

    monkeypatch.setattr(config, "tts_cache_enabled", True)
    monkeypatch.setattr(config, "redis_url", REDIS_TEST_URL)
    monkeypatch.setattr(config, "tts_provider", "fallback")
    reset_redis_cache()
    try:
        svc = tts_mod.get_tts_service()
        assert isinstance(svc, CachedTTSService)
        assert svc.provider_name.endswith("+cache")
    finally:
        reset_redis_cache()
