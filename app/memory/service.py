from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import uuid

from app.memory.neo4j_store import Neo4jConfig, Neo4jMemoryStore
from app.memory.postgres_store import PostgresConfig, PostgresMemoryStore


@dataclass
class MemoryItem:
    kind: str
    key: str
    value: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict = field(default_factory=dict)
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "memory-file"


class MemoryService:
    def __init__(self, persist_path: str | None = None) -> None:
        self._items: list[MemoryItem] = []
        self.persist_path = Path(persist_path or os.getenv("MEMORY_PERSIST_PATH", "./memory_store.json"))
        self.postgres = self._init_postgres()
        self.neo4j = self._init_neo4j()
        self._load_persisted()

    def _init_postgres(self) -> PostgresMemoryStore | None:
        dsn = os.getenv("MEMORY_POSTGRES_DSN")
        if not dsn:
            return None
        store = PostgresMemoryStore(PostgresConfig(dsn=dsn))
        try:
            store.connect()
            return store
        except Exception:
            return None

    def _init_neo4j(self) -> Neo4jMemoryStore | None:
        uri = os.getenv("MEMORY_NEO4J_URI")
        username = os.getenv("MEMORY_NEO4J_USERNAME")
        password = os.getenv("MEMORY_NEO4J_PASSWORD")
        if not (uri and username and password):
            return None
        store = Neo4jMemoryStore(Neo4jConfig(uri=uri, username=username, password=password))
        try:
            store.connect()
            return store
        except Exception:
            return None

    def _load_persisted(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
            self._items = [MemoryItem(**item) for item in data if isinstance(item, dict)]
        except Exception:
            # Keep app usable even if persisted file is corrupted.
            self._items = []

    def _flush_persisted(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
        tmp.write_text(json.dumps([asdict(i) for i in self._items], ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.persist_path)

    def _normalized(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _is_noise(self, kind: str, key: str, value: str) -> bool:
        if not key.strip() or not value.strip():
            return True
        if len(value.strip()) < 2:
            return True
        if kind == "ephemeral" and len(value.strip()) < 8:
            return True
        return False

    def _is_duplicate(self, kind: str, key: str, value: str) -> bool:
        nk = self._normalized(key)
        nv = self._normalized(value)
        for item in reversed(self._items[-200:]):
            if item.kind == kind and self._normalized(item.key) == nk and self._normalized(item.value) == nv:
                return True
        return False

    def write(self, kind: str, key: str, value: str, metadata: dict | None = None) -> MemoryItem:
        kind = (kind or "note").strip().lower()
        key = (key or "").strip()
        value = (value or "").strip()
        metadata = metadata or {}

        if self._is_noise(kind, key, value) or self._is_duplicate(kind, key, value):
            return MemoryItem(kind=kind, key=key, value=value, metadata={**metadata, "filtered": True})

        item = MemoryItem(kind=kind, key=key, value=value, metadata=metadata)
        self._items.append(item)

        created_at = None
        if self.postgres and self.postgres.available():
            created_at = self.postgres.write(kind, key, value, uid=item.uid, metadata=item.metadata)
        if self.neo4j and self.neo4j.available():
            self.neo4j.write(kind, key, value, uid=item.uid, metadata=item.metadata)

        if created_at:
            item.created_at = created_at

        self._flush_persisted()
        return item

    def search(self, query: str) -> list[MemoryItem]:
        q = (query or "").lower().strip()
        in_memory = [
            i
            for i in self._items
            if not q or q in i.key.lower() or q in i.value.lower() or q in i.kind.lower()
        ]

        pg_rows = self.postgres.search(query) if self.postgres and self.postgres.available() else []
        neo_rows = self.neo4j.search(query) if self.neo4j and self.neo4j.available() else []

        merged: dict[tuple[str, str, str], MemoryItem] = {(m.kind, m.key, m.value): m for m in in_memory}

        for row in pg_rows + neo_rows:
            key = (row["kind"], row["key"], row["value"])
            if key not in merged:
                merged[key] = MemoryItem(
                    kind=row["kind"],
                    key=row["key"],
                    value=row["value"],
                    created_at=row.get("created_at") or datetime.now(UTC).isoformat(),
                    metadata=row.get("metadata") or {},
                    uid=row.get("uid") or str(uuid.uuid4()),
                    source=row.get("source") or "db",
                )

        return list(merged.values())

    def delete(self, kind: str, key: str) -> bool:
        before = len(self._items)
        self._items = [i for i in self._items if not (i.kind == kind and i.key == key)]
        removed = len(self._items) < before

        if self.postgres and self.postgres.available():
            self.postgres.delete(kind, key)
        if self.neo4j and self.neo4j.available():
            self.neo4j.delete(kind, key)
        if removed:
            self._flush_persisted()
        return removed

    def erase_all(self) -> int:
        count = len(self._items)
        self._items = []
        if self.postgres and self.postgres.available():
            self.postgres.delete_all()
        if self.neo4j and self.neo4j.available():
            self.neo4j.delete_all()
        self._flush_persisted()
        return count
