"""Tests for the pluggable EmbeddingService.

These tests are strictly OFFLINE: only the deterministic ``"hash"`` provider is
exercised for real. The non-hash providers are verified only via their *failure*
path (they must degrade to hash), and an autouse guard hard-fails any accidental
outbound HTTP so the suite can never depend on ollama/network.
"""

from __future__ import annotations

import math

import pytest

from app.memory.embeddings import (
    DEFAULT_HASH_DIM,
    EmbeddingService,
    get_embedding_service,
    reset_embedding_service_cache,
)


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch):
    """Make any real HTTP call blow up, guaranteeing the suite stays offline."""
    try:
        import requests
    except Exception:
        return

    def _no_http(*_args, **_kwargs):  # pragma: no cover - only hit on regression
        raise AssertionError("network access is forbidden in test_embeddings")

    monkeypatch.setattr(requests, "post", _no_http, raising=False)
    monkeypatch.setattr(requests, "get", _no_http, raising=False)


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def test_hash_is_the_default_provider():
    svc = EmbeddingService()
    assert svc.provider == "hash"
    assert svc.dim == DEFAULT_HASH_DIM


def test_hash_deterministic_same_text_same_vector():
    a = EmbeddingService("hash", dim=128)
    b = EmbeddingService("hash", dim=128)
    for text in ("hello world", "你好,世界", "Mixed 中英文 text 42"):
        v1 = a.embed(text)
        v2 = a.embed(text)
        v3 = b.embed(text)
        # Stable within an instance and across instances (hashlib, not builtin hash()).
        assert v1 == v2
        assert v1 == v3


def test_hash_correct_dim_and_float_type():
    for dim in (16, 64, 256, 1024):
        svc = EmbeddingService("hash", dim=dim)
        vec = svc.embed("a memory about the user's dog")
        assert svc.dim == dim
        assert len(vec) == dim
        assert all(isinstance(x, float) for x in vec)


def test_hash_vectors_are_l2_normalized():
    svc = EmbeddingService("hash", dim=256)
    for text in ("hello world", "你好,世界 multilingual", "single", "x y z"):
        vec = svc.embed(text)
        assert abs(_l2_norm(vec) - 1.0) < 1e-6


def test_hash_empty_text_is_zero_vector_no_nan():
    svc = EmbeddingService("hash", dim=32)
    vec = svc.embed("")
    assert len(vec) == 32
    # No features -> zero vector (and crucially no divide-by-zero NaN).
    assert all(x == 0.0 for x in vec)


def test_distinct_texts_produce_distinct_vectors():
    svc = EmbeddingService("hash", dim=256)
    assert svc.embed("cat") != svc.embed("dog")


def test_batch_equals_per_item():
    svc = EmbeddingService("hash", dim=64)
    texts = ["alpha", "beta gamma", "你好世界", ""]
    batch = svc.embed_batch(texts)
    assert batch == [svc.embed(t) for t in texts]
    assert len(batch) == len(texts)
    assert all(len(v) == 64 for v in batch)


def test_unknown_provider_degrades_to_hash():
    ref = EmbeddingService("hash", dim=256)
    unknown = EmbeddingService("totally-not-a-real-provider", dim=256)
    for text in ("hello", "你好世界", "some memory text"):
        out = unknown.embed(text)
        assert out == ref.embed(text)
        assert len(out) == 256


def test_failed_ollama_provider_degrades_to_hash(monkeypatch):
    # Simulate a backend error WITHOUT touching the network.
    svc = EmbeddingService("ollama", model="bge-m3", dim=256)
    ref = EmbeddingService("hash", dim=256)

    def _boom(_text):
        raise RuntimeError("simulated ollama outage")

    monkeypatch.setattr(svc, "_embed_ollama", _boom)
    for text in ("hello", "你好"):
        assert svc.embed(text) == ref.embed(text)


def test_ollama_wrong_dimension_degrades_to_hash(monkeypatch):
    # Backend "succeeds" but returns the wrong length -> the dim invariant forces fallback.
    svc = EmbeddingService("ollama", model="bge-m3", dim=256)
    ref = EmbeddingService("hash", dim=256)
    monkeypatch.setattr(svc, "_embed_ollama", lambda _t: [0.123] * 999)
    assert svc.embed("hello") == ref.embed("hello")
    assert len(svc.embed("hello")) == 256


def test_failed_sentence_transformers_degrades_to_hash(monkeypatch):
    svc = EmbeddingService("sentence_transformers", model="bge-m3", dim=256)
    ref = EmbeddingService("hash", dim=256)
    monkeypatch.setattr(
        svc,
        "_embed_sentence_transformers",
        lambda _t: (_ for _ in ()).throw(ImportError("no torch here")),
    )
    assert svc.embed("hello world") == ref.embed("hello world")


def test_factory_default_is_hash_and_offline():
    reset_embedding_service_cache()
    svc = get_embedding_service()  # reads app.config; semantic layer is inert/hash by default
    assert svc.provider == "hash"
    vec = svc.embed("hello from the factory")
    assert len(vec) == svc.dim
    assert abs(_l2_norm(vec) - 1.0) < 1e-6


def test_factory_resolves_provider_and_auto_dim_from_config():
    reset_embedding_service_cache()

    class _Cfg:
        embedding_provider = "ollama"
        embedding_model = "bge-m3"
        embedding_dim = 0  # auto
        ollama_base_url = "http://127.0.0.1:11434"

    # Construction only: no embed() call, so no network.
    svc = get_embedding_service(_Cfg())
    assert svc.provider == "ollama"
    assert svc.model == "bge-m3"
    assert svc.dim == 1024  # auto-resolved for bge-m3
    reset_embedding_service_cache()
