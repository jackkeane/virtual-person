from __future__ import annotations

from pathlib import Path

from app.memory.service import MemoryService


def test_memory_filters_noise_and_duplicate(tmp_path: Path):
    svc = MemoryService(persist_path=str(tmp_path / "mem.json"))

    n1 = svc.write("note", "", "x")
    assert n1.metadata.get("filtered") is True

    ok = svc.write("identity", "name", "Liyang")
    assert ok.metadata.get("filtered") is None

    dup = svc.write("identity", "name", "Liyang")
    assert dup.metadata.get("filtered") is True


def test_memory_persistence_survives_restart(tmp_path: Path):
    persist = tmp_path / "durable.json"

    svc1 = MemoryService(persist_path=str(persist))
    svc1.write("identity", "name", "RestartUser", metadata={"source": "test"})

    # Simulate app restart by constructing a new service instance.
    svc2 = MemoryService(persist_path=str(persist))
    hits = svc2.search("RestartUser")
    assert any(i.key == "name" and i.value == "RestartUser" for i in hits)


def test_memory_erase_all(tmp_path: Path):
    svc = MemoryService(persist_path=str(tmp_path / "erase.json"))
    svc.write("note", "k1", "v1")
    svc.write("note", "k2", "v2")
    removed = svc.erase_all()
    assert removed >= 2
    assert svc.search("") == []
