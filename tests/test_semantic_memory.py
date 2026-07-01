"""Tests for the gated semantic-retrieval layer of MemoryService.

Everything here is strictly OFFLINE and DETERMINISTIC: the default ``"hash"``
embedder (pure python + numpy) and the in-process ``NumpyVectorIndex`` fallback
are exercised, so there is no network / Redis / Postgres / pgvector / torch /
ollama dependency. The pgvector/ollama-backed paths are covered elsewhere and
skip cleanly when their backends are absent.

The through-line of the file is the GATE: with ``semantic_memory_enabled`` False
(the default) MemoryService must behave exactly like the pre-existing keyword /
curation path; only when the flag is explicitly enabled does the hybrid semantic
ranking activate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import config
from app.memory.curation import retrieval_score
from app.memory.embeddings import reset_embedding_service_cache
from app.memory.service import MemoryService


def _enable_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the semantic layer on with the deterministic offline hash embedder.

    ``pgvector_dsn`` is forced empty so the store is always the in-process
    ``NumpyVectorIndex`` — no DB, fully reproducible.
    """
    monkeypatch.setattr(config, "semantic_memory_enabled", True)
    monkeypatch.setattr(config, "embedding_provider", "hash")
    monkeypatch.setattr(config, "pgvector_dsn", "")
    reset_embedding_service_cache()


# --------------------------------------------------------------------------- #
# GATE OFF (default): the semantic layer is completely inert.
# --------------------------------------------------------------------------- #
def test_semantic_is_inert_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Pin the default explicitly so ordering with other tests can never matter.
    monkeypatch.setattr(config, "semantic_memory_enabled", False)

    svc = MemoryService(persist_path=str(tmp_path / "mem.json"))

    # Nothing was built — the service is byte-identical to the keyword path.
    assert svc._embedder is None
    assert svc._vector is None
    assert svc._semantic_ready is False
    assert svc._semantic_active() is False

    svc.write("identity", "name", "Liyang")
    svc.write("note", "story", "the quick brown fox jumps over the lazy dog")

    # Plain substring keyword matching, unchanged.
    assert [h.value for h in svc.search("quick brown")] == [
        "the quick brown fox jumps over the lazy dog"
    ]
    # A non-contiguous multi-word query does NOT substring-match -> keyword miss.
    assert svc.search("quick fox lazy") == []


def test_disabled_search_matches_plain_keyword_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "semantic_memory_enabled", False)
    svc = MemoryService(persist_path=str(tmp_path / "mem.json"))
    svc.write("identity", "name", "Liyang")
    svc.write("preference", "food", "loves spicy sichuan hotpot")
    svc.write("note", "pet", "has a cat named Mochi")

    q = "cat"
    got = {(h.kind, h.key, h.value) for h in svc.search(q)}
    # Reference: exactly the in-memory substring filter search() has always used.
    expected = {
        (i.kind, i.key, i.value)
        for i in svc._items
        if q in i.key.lower() or q in i.value.lower() or q in i.kind.lower()
    }
    assert got == expected
    assert expected == {("note", "pet", "has a cat named Mochi")}


# --------------------------------------------------------------------------- #
# GATE ON: writes get embedded + indexed.
# --------------------------------------------------------------------------- #
def test_enabled_indexes_every_accepted_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_semantic(monkeypatch)
    svc = MemoryService(persist_path=str(tmp_path / "mem.json"))

    assert svc._semantic_ready is True
    assert svc._embedder is not None
    assert svc._vector is not None
    assert svc._semantic_active() is True

    svc.write("identity", "name", "Liyang")
    svc.write("note", "story", "the quick brown fox jumps over the lazy dog")
    svc.write("preference", "food", "loves spicy sichuan hotpot")

    # Every accepted write was embedded + upserted into the vector index.
    assert len(svc._vector) == 3


def test_filtered_items_are_never_embedded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_semantic(monkeypatch)
    svc = MemoryService(persist_path=str(tmp_path / "mem.json"))

    noise = svc.write("note", "", "x")  # empty key -> noise, filtered
    assert noise.metadata.get("filtered") is True
    assert len(svc._vector) == 0  # noise never reaches the index

    svc.write("identity", "name", "Liyang")
    assert len(svc._vector) == 1

    dup = svc.write("identity", "name", "Liyang")  # duplicate -> filtered
    assert dup.metadata.get("filtered") is True
    assert len(svc._vector) == 1  # duplicate not re-embedded


# --------------------------------------------------------------------------- #
# GATE ON: semantic recall reaches memories the keyword filter misses.
# --------------------------------------------------------------------------- #
def test_semantic_recall_beyond_keyword(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_semantic(monkeypatch)
    svc = MemoryService(persist_path=str(tmp_path / "mem.json"))
    svc.write("note", "story", "the quick brown fox jumps over the lazy dog")

    query = "quick fox lazy dog"  # not a contiguous substring anywhere

    # Same service, gate ON: the hash embedder's word/char-ngram overlap surfaces
    # the memory that the substring filter cannot reach.
    hits = svc.search(query)
    assert any(h.key == "story" for h in hits)

    # And with the gate OFF the identical query is a keyword miss -> the crisp
    # gating contrast (recall exists only when semantic memory is enabled).
    monkeypatch.setattr(config, "semantic_memory_enabled", False)
    off = MemoryService(persist_path=str(tmp_path / "mem.json"))  # reloads same items
    assert off.search(query) == []


# --------------------------------------------------------------------------- #
# GATE ON: the blend actually re-orders results (isolates the semantic signal).
# --------------------------------------------------------------------------- #
def test_blend_flips_a_pure_curation_ordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_semantic(monkeypatch)
    svc = MemoryService(persist_path=str(tmp_path / "mem.json"))

    # A high-importance identity that is TOPICALLY unrelated to the query, plus a
    # low-importance note that is topically ON-point.
    name_item = svc.write("identity", "name", "张三")
    fact_item = svc.write("note", "fact", "the quick brown fox jumps over the lazy dog")

    query = "quick fox lazy dog"  # keyword-misses both (non-contiguous)

    # Curation ALONE would rank the identity first (kind weight 1.0 > note 0.4)...
    name_cur = retrieval_score("identity", "name", "张三", name_item.created_at, query)
    fact_cur = retrieval_score("note", "fact", fact_item.value, fact_item.created_at, query)
    assert name_cur > fact_cur

    # ...but the semantic blend surfaces the topically-relevant note ahead of it.
    ranked = svc.search(query)
    keys = [r.key for r in ranked]
    assert keys[0] == "fact"
    assert "name" in keys  # recall is preserved, just ranked lower
    assert keys.index("fact") < keys.index("name")


def test_more_similar_memory_outranks_less_similar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_semantic(monkeypatch)
    svc = MemoryService(persist_path=str(tmp_path / "mem.json"))
    svc.write("note", "near", "the quick brown fox jumps over the lazy dog")
    svc.write("note", "far", "unrelated banana smoothie recipe with mango")

    ranked = svc.search("quick brown fox lazy dog")
    keys = [r.key for r in ranked]
    assert keys[0] == "near"  # higher cosine + higher overlap -> ranked first
    assert keys.index("near") < keys.index("far")


# --------------------------------------------------------------------------- #
# GATE ON: any semantic error fails soft to the keyword result (never breaks).
# --------------------------------------------------------------------------- #
def test_semantic_failure_falls_back_to_keyword(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_semantic(monkeypatch)
    svc = MemoryService(persist_path=str(tmp_path / "mem.json"))
    svc.write("note", "story", "the quick brown fox jumps over the lazy dog")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("embedder exploded mid-turn")

    # Break the query-time embed AFTER indexing; the turn must still succeed using
    # the keyword result rather than raising.
    monkeypatch.setattr(svc._embedder, "embed", _boom)

    hits = svc.search("quick brown")  # keyword substring still matches
    assert [h.value for h in hits] == ["the quick brown fox jumps over the lazy dog"]


# --------------------------------------------------------------------------- #
# GATE ON: delete / erase_all keep the vector index consistent.
# --------------------------------------------------------------------------- #
def test_delete_and_erase_all_prune_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_semantic(monkeypatch)
    svc = MemoryService(persist_path=str(tmp_path / "mem.json"))
    svc.write("note", "a", "alpha memory about apples")
    svc.write("note", "b", "beta memory about bananas")
    assert len(svc._vector) == 2

    assert svc.delete("note", "a") is True
    assert len(svc._vector) == 1  # deleted item pruned from the index

    svc.erase_all()
    assert len(svc._vector) == 0
