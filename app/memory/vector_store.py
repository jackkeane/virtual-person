"""Semantic vector storage for the memory layer.

Two interchangeable backends behind one small interface:

    upsert(id, text, vector, metadata) -> None
    query(vector, top_k) -> list[(id, score, metadata)]   # score = cosine similarity

* ``NumpyVectorIndex`` — pure in-process numpy cosine index. Needs nothing beyond
  numpy, is ALWAYS available, and powers the local demo plus the default test path.
* ``PgVectorStore`` — real pgvector via psycopg (already a dependency), raw SQL over
  its OWN table ``vp_mem_vectors``. It connects ONLY through ``config.pgvector_dsn``
  (env ``PGVECTOR_DSN``) and NEVER touches the primary ``memory_items`` table /
  ``MEMORY_POSTGRES_DSN``. When the dsn is empty or the connection / ``vector``
  extension is unavailable it is simply absent — the factory returns ``None`` and
  callers fall back to ``NumpyVectorIndex``.

This module is INERT by default: nothing here imports psycopg at module load, and
``get_vector_store`` returns ``None`` whenever ``PGVECTOR_DSN`` is unset, so a plain
``pytest`` run (semantic layer disabled) behaves exactly as before.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
import json
from typing import Optional, Sequence

import numpy as np

# Our OWN table — deliberately separate from the primary `memory_items` table.
VP_MEM_VECTORS_TABLE = "vp_mem_vectors"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_vector(vector: Sequence[float] | np.ndarray, dim: int) -> np.ndarray:
    """Coerce ``vector`` to a 1-D float64 array of exactly ``dim`` elements."""
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"expected vector of dim {dim}, got {arr.shape[0]}")
    return arr


def _to_pgvector_literal(vector: Sequence[float] | np.ndarray, dim: int) -> str:
    """Render ``vector`` as a pgvector text literal ``[f1,f2,...]`` (cast ``::vector``)."""
    arr = _as_vector(vector, dim)
    return "[" + ",".join(str(v) for v in arr.tolist()) + "]"


def _cosine_similarities(matrix: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of ``q`` against each row of ``matrix``.

    Zero-norm rows (or a zero query) yield 0.0 similarity rather than NaN.
    """
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0 or matrix.shape[0] == 0:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * q_norm
    dots = matrix @ q
    return np.divide(
        dots,
        denom,
        out=np.zeros_like(dots, dtype=np.float64),
        where=denom > 0,
    )


# --------------------------------------------------------------------------- #
# shared interface
# --------------------------------------------------------------------------- #
class VectorStore(abc.ABC):
    """Shared cosine-similarity vector store interface."""

    @abc.abstractmethod
    def upsert(
        self,
        id: str,
        text: str,
        vector: Sequence[float] | np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """Insert or replace the record ``id`` with ``vector`` and its payload."""

    @abc.abstractmethod
    def query(
        self,
        vector: Sequence[float] | np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict]]:
        """Return the ``top_k`` nearest records as ``(id, cosine_similarity, metadata)``."""

    def available(self) -> bool:  # overridden by backends that can be absent
        return True


# --------------------------------------------------------------------------- #
# in-process numpy backend (always available)
# --------------------------------------------------------------------------- #
class NumpyVectorIndex(VectorStore):
    """In-process cosine index backed by a numpy matrix. Deterministic, dependency-light."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self._pos: dict[str, int] = {}
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id: str) -> bool:
        return id in self._pos

    def available(self) -> bool:
        return True

    def upsert(
        self,
        id: str,
        text: str,
        vector: Sequence[float] | np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        vec = _as_vector(vector, self.dim)
        meta = dict(metadata or {})
        if id in self._pos:
            i = self._pos[id]
            self._vectors[i] = vec
            self._texts[i] = text
            self._metadata[i] = meta
        else:
            self._pos[id] = len(self._ids)
            self._ids.append(id)
            self._vectors.append(vec)
            self._texts.append(text)
            self._metadata.append(meta)

    def query(
        self,
        vector: Sequence[float] | np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict]]:
        if not self._ids or top_k <= 0:
            return []
        q = _as_vector(vector, self.dim)
        matrix = np.vstack(self._vectors)  # (n, dim)
        sims = _cosine_similarities(matrix, q)  # (n,)
        k = min(top_k, len(self._ids))
        # Stable sort on the negated similarity => ties keep insertion order,
        # which makes the top-k fully deterministic.
        order = np.argsort(-sims, kind="stable")[:k]
        return [
            (self._ids[i], float(sims[i]), dict(self._metadata[i]))
            for i in order
        ]

    def delete(self, id: str) -> None:
        i = self._pos.pop(id, None)
        if i is None:
            return
        del self._ids[i]
        del self._vectors[i]
        del self._texts[i]
        del self._metadata[i]
        # Reindex positions after the removed slot.
        for pos, key in enumerate(self._ids):
            self._pos[key] = pos

    def delete_all(self) -> None:
        self._pos.clear()
        self._ids.clear()
        self._vectors.clear()
        self._texts.clear()
        self._metadata.clear()


# --------------------------------------------------------------------------- #
# pgvector backend (optional, never raises out of the factory)
# --------------------------------------------------------------------------- #
@dataclass
class PgVectorConfig:
    dsn: str
    dim: int
    table: str = VP_MEM_VECTORS_TABLE


class PgVectorStore(VectorStore):
    """Real pgvector store over ``vp_mem_vectors``. Mirrors PostgresMemoryStore shape."""

    def __init__(self, cfg: PgVectorConfig) -> None:
        self.cfg = cfg
        self._conn = None

    def connect(self) -> None:
        import psycopg  # lazy: keeps import-time deps clean, like PostgresMemoryStore

        conn = psycopg.connect(self.cfg.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.cfg.table} (
                        id TEXT PRIMARY KEY,
                        content TEXT,
                        embedding vector({int(self.cfg.dim)}),
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
            conn.commit()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise
        self._conn = conn

    def available(self) -> bool:
        return self._conn is not None

    def upsert(
        self,
        id: str,
        text: str,
        vector: Sequence[float] | np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        if not self._conn:
            return
        literal = _to_pgvector_literal(vector, self.cfg.dim)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.cfg.table} (id, content, embedding, metadata)
                VALUES (%s, %s, %s::vector, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                  content = EXCLUDED.content,
                  embedding = EXCLUDED.embedding,
                  metadata = EXCLUDED.metadata
                """,
                (id, text, literal, json.dumps(metadata or {})),
            )
        self._conn.commit()

    def query(
        self,
        vector: Sequence[float] | np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict]]:
        if not self._conn or top_k <= 0:
            return []
        literal = _to_pgvector_literal(vector, self.cfg.dim)
        with self._conn.cursor() as cur:
            # `<=>` is pgvector cosine DISTANCE; 1 - distance = cosine similarity.
            cur.execute(
                f"""
                SELECT id, 1 - (embedding <=> %s::vector) AS score, metadata::text
                FROM {self.cfg.table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (literal, literal, int(top_k)),
            )
            rows = cur.fetchall()
        return [(r[0], float(r[1]), json.loads(r[2] or "{}")) for r in rows]

    def delete(self, id: str) -> None:
        if not self._conn:
            return
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.cfg.table} WHERE id = %s", (id,))
        self._conn.commit()

    def delete_all(self) -> None:
        if not self._conn:
            return
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.cfg.table}")
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def get_vector_store(dim: int, dsn: Optional[str] = None) -> Optional[PgVectorStore]:
    """Return a connected ``PgVectorStore`` when pgvector is usable, else ``None``.

    Reads ``config.pgvector_dsn`` (env ``PGVECTOR_DSN``) unless an explicit ``dsn`` is
    given. Empty dsn, an unreachable server, or a missing ``vector`` extension all
    yield ``None`` — this function NEVER raises. Callers use ``NumpyVectorIndex`` as
    the fallback when ``None`` is returned.
    """
    from app.config import config

    dsn = config.pgvector_dsn if dsn is None else dsn
    if not dsn:
        return None
    store = PgVectorStore(PgVectorConfig(dsn=dsn, dim=int(dim)))
    try:
        store.connect()
    except Exception:
        return None
    return store if store.available() else None
