from __future__ import annotations

import os

import numpy as np
import pytest

from app.config import config
from app.memory.vector_store import (
    NumpyVectorIndex,
    PgVectorConfig,
    PgVectorStore,
    VectorStore,
    get_vector_store,
)

PGVECTOR_DSN = os.getenv("PGVECTOR_DSN")


# --------------------------------------------------------------------------- #
# NumpyVectorIndex — ALWAYS runs (numpy only, no network / db / torch / ollama)
# --------------------------------------------------------------------------- #
def test_both_backends_share_the_interface() -> None:
    assert issubclass(NumpyVectorIndex, VectorStore)
    assert issubclass(PgVectorStore, VectorStore)


def test_upsert_and_query_returns_nearest_by_cosine() -> None:
    idx = NumpyVectorIndex(dim=3)
    idx.upsert("a", "apple", [1.0, 0.0, 0.0], {"k": "a"})
    idx.upsert("b", "banana", [0.0, 1.0, 0.0], {"k": "b"})
    idx.upsert("c", "cherry", [0.0, 0.0, 1.0], {"k": "c"})

    res = idx.query([0.9, 0.1, 0.0], top_k=2)

    assert [r[0] for r in res] == ["a", "b"]  # nearest first
    assert res[0][1] > res[1][1]  # scores strictly descending
    assert res[0][2] == {"k": "a"}  # metadata carried through
    # cosine([0.9,0.1,0],[1,0,0]) ~= 0.9939
    assert res[0][1] == pytest.approx(0.9 / np.sqrt(0.82))


def test_cosine_orientation_aligned_orthogonal_opposite() -> None:
    idx = NumpyVectorIndex(dim=2)
    idx.upsert("same", "", [2.0, 0.0])  # aligned (magnitude irrelevant to cosine)
    idx.upsert("orth", "", [0.0, 5.0])  # orthogonal
    idx.upsert("opp", "", [-3.0, 0.0])  # opposite

    res = idx.query([1.0, 0.0], top_k=3)
    by_id = {r[0]: r[1] for r in res}

    assert [r[0] for r in res] == ["same", "orth", "opp"]
    assert by_id["same"] == pytest.approx(1.0)
    assert by_id["orth"] == pytest.approx(0.0)
    assert by_id["opp"] == pytest.approx(-1.0)


def test_query_is_deterministic() -> None:
    idx = NumpyVectorIndex(dim=4)
    vectors = {
        "v1": [0.2, 0.9, 0.1, 0.0],
        "v2": [0.8, 0.1, 0.3, 0.4],
        "v3": [0.1, 0.1, 0.9, 0.2],
        "v4": [0.5, 0.5, 0.5, 0.5],
        "v5": [0.0, 0.0, 0.1, 1.0],
    }
    for k, v in vectors.items():
        idx.upsert(k, k, v, {"id": k})

    q = [0.3, 0.7, 0.2, 0.1]
    first = idx.query(q, top_k=3)
    second = idx.query(q, top_k=3)

    assert first == second  # identical results across repeated calls


def test_ties_break_on_insertion_order() -> None:
    idx = NumpyVectorIndex(dim=2)
    idx.upsert("first", "", [1.0, 0.0])
    idx.upsert("second", "", [1.0, 0.0])  # identical vector -> tie

    res = idx.query([1.0, 0.0], top_k=2)

    assert [r[0] for r in res] == ["first", "second"]  # stable insertion order


def test_upsert_overwrites_existing_id() -> None:
    idx = NumpyVectorIndex(dim=3)
    idx.upsert("a", "apple", [1.0, 0.0, 0.0], {"v": 1})
    idx.upsert("a", "apricot", [0.0, 1.0, 0.0], {"v": 2})

    assert len(idx) == 1  # replaced, not duplicated
    res = idx.query([0.0, 1.0, 0.0], top_k=1)
    assert res[0][0] == "a"
    assert res[0][1] == pytest.approx(1.0)
    assert res[0][2] == {"v": 2}


def test_top_k_limits_results() -> None:
    idx = NumpyVectorIndex(dim=3)
    idx.upsert("a", "", [1.0, 0.0, 0.0])
    idx.upsert("b", "", [0.0, 1.0, 0.0])
    idx.upsert("c", "", [0.0, 0.0, 1.0])

    assert len(idx.query([1.0, 0.0, 0.0], top_k=2)) == 2
    assert len(idx.query([1.0, 0.0, 0.0], top_k=10)) == 3  # capped at index size
    assert idx.query([1.0, 0.0, 0.0], top_k=0) == []


def test_empty_index_returns_empty() -> None:
    idx = NumpyVectorIndex(dim=3)
    assert idx.query([1.0, 0.0, 0.0]) == []
    assert len(idx) == 0


def test_metadata_is_copied_out() -> None:
    idx = NumpyVectorIndex(dim=3)
    idx.upsert("a", "t", [1.0, 0.0, 0.0], {"kind": "identity", "key": "name"})

    res = idx.query([1.0, 0.0, 0.0], top_k=1)
    assert res[0][2] == {"kind": "identity", "key": "name"}

    # Mutating the returned dict must not corrupt the stored record.
    res[0][2]["kind"] = "tampered"
    again = idx.query([1.0, 0.0, 0.0], top_k=1)
    assert again[0][2]["kind"] == "identity"


def test_delete_and_delete_all() -> None:
    idx = NumpyVectorIndex(dim=3)
    idx.upsert("a", "", [1.0, 0.0, 0.0])
    idx.upsert("b", "", [0.0, 1.0, 0.0])
    idx.upsert("c", "", [0.0, 0.0, 1.0])

    idx.delete("b")
    assert "b" not in idx
    ids = {r[0] for r in idx.query([1.0, 1.0, 1.0], top_k=5)}
    assert ids == {"a", "c"}
    # remaining records still queryable and correct
    assert idx.query([0.0, 0.0, 1.0], top_k=1)[0][0] == "c"

    idx.delete_all()
    assert len(idx) == 0
    assert idx.query([1.0, 0.0, 0.0]) == []


def test_dim_mismatch_raises() -> None:
    idx = NumpyVectorIndex(dim=3)
    with pytest.raises(ValueError):
        idx.upsert("a", "t", [1.0, 0.0])  # too short
    idx.upsert("a", "t", [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        idx.query([1.0, 0.0, 0.0, 0.0])  # too long


def test_zero_vector_scores_zero_not_nan() -> None:
    idx = NumpyVectorIndex(dim=3)
    idx.upsert("z", "", [0.0, 0.0, 0.0])
    idx.upsert("a", "", [1.0, 0.0, 0.0])
    res = idx.query([1.0, 0.0, 0.0], top_k=2)
    scores = {r[0]: r[1] for r in res}
    assert scores["z"] == 0.0  # no NaN from a zero-norm row
    assert scores["a"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# get_vector_store factory — ALWAYS runs; inert by default, never raises
# --------------------------------------------------------------------------- #
def test_get_vector_store_returns_none_when_dsn_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "pgvector_dsn", "")
    assert get_vector_store(8) is None  # -> caller falls back to NumpyVectorIndex


def test_get_vector_store_returns_none_on_unreachable_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A refused localhost port: connect fails fast, factory swallows it -> None.
    monkeypatch.setattr(
        config, "pgvector_dsn", "postgresql://127.0.0.1:1/none?connect_timeout=1"
    )
    assert get_vector_store(8) is None  # never raises


# --------------------------------------------------------------------------- #
# PgVectorStore — SKIPS unless a real pgvector is provided via PGVECTOR_DSN
# (runs in CI against pgvector/pgvector:pg16; never touches the main postgres).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not PGVECTOR_DSN, reason="no pgvector (PGVECTOR_DSN unset)")
def test_pgvector_store_roundtrip() -> None:
    dim = 4
    store = get_vector_store(dim, dsn=PGVECTOR_DSN)
    if store is None:
        pytest.skip("PGVECTOR_DSN set but pgvector unavailable (connect/extension failed)")
    try:
        assert store.available()
        store.delete_all()  # isolate: our own vp_mem_vectors table only

        store.upsert("a", "apple", [1.0, 0.0, 0.0, 0.0], {"k": "a"})
        store.upsert("b", "banana", [0.0, 1.0, 0.0, 0.0], {"k": "b"})
        store.upsert("c", "cherry", [0.0, 0.0, 1.0, 0.0], {"k": "c"})

        res = store.query([0.9, 0.1, 0.0, 0.0], top_k=2)
        assert len(res) == 2
        assert res[0][0] == "a"  # nearest by cosine
        assert res[0][1] > res[1][1]
        assert res[0][2] == {"k": "a"}
        assert 0.99 < res[0][1] <= 1.0001  # cosine similarity, float4 tolerance

        # ON CONFLICT upsert replaces content + embedding + metadata.
        store.upsert("a", "apricot", [0.0, 0.0, 0.0, 1.0], {"k": "a2"})
        r2 = store.query([0.0, 0.0, 0.0, 1.0], top_k=1)
        assert r2[0][0] == "a"
        assert r2[0][2] == {"k": "a2"}

        store.delete("a")
        remaining = [r[0] for r in store.query([0.0, 0.0, 0.0, 1.0], top_k=5)]
        assert "a" not in remaining
    finally:
        try:
            store.delete_all()
        except Exception:
            pass
        store.close()


@pytest.mark.skipif(not PGVECTOR_DSN, reason="no pgvector (PGVECTOR_DSN unset)")
def test_pgvector_factory_config_uses_own_table() -> None:
    # The pgvector store must live on its OWN table, never `memory_items`.
    assert PgVectorConfig(dsn="x", dim=4).table == "vp_mem_vectors"
    store = get_vector_store(4, dsn=PGVECTOR_DSN)
    if store is None:
        pytest.skip("PGVECTOR_DSN set but pgvector unavailable")
    try:
        assert store.cfg.table == "vp_mem_vectors"
    finally:
        store.close()
