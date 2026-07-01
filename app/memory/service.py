from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import uuid

from app.config import config
from app.memory.neo4j_store import Neo4jConfig, Neo4jMemoryStore
from app.memory.postgres_store import PostgresConfig, PostgresMemoryStore

# Weight of the semantic cosine similarity vs. the curation score when the gated
# hybrid retrieval path blends the two (0 = pure curation, 1 = pure semantic).
# Held at 0.5 so neither signal dominates. Consulted ONLY when the semantic layer
# is enabled AND ready; the default keyword/curation path never reads it.
_SEMANTIC_BLEND_ALPHA = 0.5


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

        # Use a stable absolute default path so restarts from different cwd don't "lose" memory.
        default_persist = Path(__file__).resolve().parents[2] / "memory_store.json"
        self.persist_path = Path(persist_path or os.getenv("MEMORY_PERSIST_PATH", str(default_persist)))

        self.postgres = self._init_postgres()
        self.neo4j = self._init_neo4j()

        # Semantic layer: inert by default. These stay None unless
        # config.semantic_memory_enabled is true AND the embedder/index build
        # succeeds (see _init_semantic). When they are None every semantic hook
        # below is skipped and the service behaves exactly like the keyword path.
        self._embedder = None
        self._vector = None
        self._semantic_ready = False

        self._load_persisted()
        self._init_semantic()

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

    # ------------------------------------------------------------------ #
    # Semantic layer (gated, fail-soft, inert by default)
    # ------------------------------------------------------------------ #
    def _init_semantic(self) -> None:
        """Build the embedder + vector index when semantic memory is enabled.

        Mirrors _init_postgres: on *any* problem (flag off, import error, connect
        failure) the semantic layer stays fully off (embedder/vector None,
        _semantic_ready False) and the service is byte-identical to the keyword/
        curation path. Heavy deps (numpy, via the embedding/vector modules) are
        imported lazily HERE so a default run never imports them.
        """
        if not config.semantic_memory_enabled:
            return
        try:
            from app.memory.embeddings import get_embedding_service
            from app.memory.vector_store import NumpyVectorIndex, get_vector_store

            embedder = get_embedding_service()
            dim = int(getattr(embedder, "dim", 0)) or 256

            # Prefer a real pgvector store (its OWN table/DSN, never memory_items);
            # otherwise use a process-local numpy cosine index seeded from the
            # already-loaded items.
            vector = get_vector_store(dim)
            seed_from_items = False
            if vector is None:
                vector = NumpyVectorIndex(dim=dim)
                seed_from_items = True

            self._embedder = embedder
            self._vector = vector
            self._semantic_ready = True
            if seed_from_items:
                self._reindex_all()
        except Exception:
            self._embedder = None
            self._vector = None
            self._semantic_ready = False

    def _semantic_active(self) -> bool:
        return (
            config.semantic_memory_enabled
            and self._semantic_ready
            and self._vector is not None
            and self._embedder is not None
        )

    def _index_item(self, item: MemoryItem) -> None:
        """Embed + upsert one item into the vector index. Never raises."""
        if not self._semantic_active():
            return
        try:
            text = f"{item.key} {item.value}".strip()
            self._vector.upsert(
                item.uid,
                text,
                self._embedder.embed(text),
                {
                    "kind": item.kind,
                    "key": item.key,
                    "value": item.value,
                    "created_at": item.created_at,
                    "uid": item.uid,
                },
            )
        except Exception:
            # An embedding/index error must never break a write or app startup.
            pass

    def _reindex_all(self) -> None:
        """Seed the (process-local) vector index from the current items."""
        if not self._semantic_active():
            return
        for item in self._items:
            self._index_item(item)

    def _semantic_search(self, query: str, keyword_items: list[MemoryItem]) -> list[MemoryItem] | None:
        """Hybrid re-rank blending cosine similarity with the curation score.

        Returns a ranked list (keyword hits UNION semantic top-k), or None to
        signal 'use the keyword result unchanged' (layer off, empty query, or any
        failure). Semantic recall never drops keyword hits — they stay candidates
        with at least their own similarity (default 0.0).
        """
        if not self._semantic_active():
            return None
        q = (query or "").strip()
        if not q:
            return None
        try:
            from app.memory.curation import retrieval_score

            top_k = int(getattr(config, "semantic_top_k", 5) or 5)
            hits = self._vector.query(self._embedder.embed(q), top_k=top_k)
        except Exception:
            return None

        sim_by_uid: dict[str, float] = {}
        for uid, sim, _meta in hits:
            if uid not in sim_by_uid or sim > sim_by_uid[uid]:
                sim_by_uid[uid] = float(sim)

        items_by_uid = {it.uid: it for it in self._items}

        # Candidate set keyed by (kind, key, value) — the dedup key used everywhere.
        candidates: dict[tuple[str, str, str], tuple[MemoryItem, float]] = {}

        def _add(item: MemoryItem, sim: float) -> None:
            k = (item.kind, item.key, item.value)
            prev = candidates.get(k)
            if prev is None or sim > prev[1]:
                candidates[k] = (item, sim)

        for it in keyword_items:
            _add(it, sim_by_uid.get(it.uid, 0.0))
        for uid, sim in sim_by_uid.items():
            it = items_by_uid.get(uid)
            if it is not None:
                _add(it, sim)

        if not candidates:
            return None

        def _blended(item: MemoryItem, sim: float) -> float:
            curation = retrieval_score(item.kind, item.key, item.value, item.created_at, query)
            curation_norm = min(max(curation / 1.5, 0.0), 1.0)
            sim_norm = min(max(sim, 0.0), 1.0)
            return _SEMANTIC_BLEND_ALPHA * sim_norm + (1.0 - _SEMANTIC_BLEND_ALPHA) * curation_norm

        ranked = sorted(candidates.values(), key=lambda pair: _blended(pair[0], pair[1]), reverse=True)
        return [item for item, _ in ranked]

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

        # Gated semantic indexing. Reached only for ACCEPTED items — noise/duplicate
        # items returned early above, so filtered items are never embedded. Fail-soft.
        if self._semantic_active():
            self._index_item(item)

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

        results = list(merged.values())

        # Gated hybrid semantic re-ranking. INERT by default: when semantic memory
        # is disabled (or unavailable) this block is skipped and the keyword/curation
        # result is returned byte-identically to before. Fail-soft — any semantic
        # error falls back to the keyword result and never breaks a turn.
        if self._semantic_active():
            try:
                ranked = self._semantic_search(query, results)
                if ranked is not None:
                    return ranked
            except Exception:
                pass

        return results

    def delete(self, kind: str, key: str) -> bool:
        before = len(self._items)
        removed_items = [i for i in self._items if (i.kind == kind and i.key == key)]
        self._items = [i for i in self._items if not (i.kind == kind and i.key == key)]
        removed = len(self._items) < before

        if self.postgres and self.postgres.available():
            self.postgres.delete(kind, key)
        if self.neo4j and self.neo4j.available():
            self.neo4j.delete(kind, key)
        if self._semantic_active():
            for it in removed_items:
                try:
                    self._vector.delete(it.uid)
                except Exception:
                    pass
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
        if self._semantic_active():
            try:
                self._vector.delete_all()
            except Exception:
                pass
        self._flush_persisted()
        return count
