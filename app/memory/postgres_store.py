from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class PostgresConfig:
    dsn: str


class PostgresMemoryStore:
    def __init__(self, cfg: PostgresConfig) -> None:
        self.cfg = cfg
        self._conn = None

    def connect(self) -> None:
        import psycopg

        self._conn = psycopg.connect(self.cfg.dsn)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id BIGSERIAL PRIMARY KEY,
                    uid TEXT UNIQUE,
                    kind TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute("ALTER TABLE memory_items ADD COLUMN IF NOT EXISTS uid TEXT")
            cur.execute("ALTER TABLE memory_items ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_uid ON memory_items(uid)")
        self._conn.commit()

    def available(self) -> bool:
        return self._conn is not None

    def write(self, kind: str, key: str, value: str, uid: str | None = None, metadata: dict | None = None) -> Optional[str]:
        if not self._conn:
            return None
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_items (uid, kind, key, value, metadata)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (uid) DO UPDATE SET
                  kind = EXCLUDED.kind,
                  key = EXCLUDED.key,
                  value = EXCLUDED.value,
                  metadata = EXCLUDED.metadata
                RETURNING created_at::text
                """,
                (uid, kind, key, value, json.dumps(metadata or {})),
            )
            row = cur.fetchone()
        self._conn.commit()
        return row[0] if row else None

    def delete(self, kind: str, key: str) -> None:
        if not self._conn:
            return
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM memory_items WHERE kind = %s AND key = %s", (kind, key))
        self._conn.commit()

    def delete_all(self) -> None:
        if not self._conn:
            return
        with self._conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE memory_items")
        self._conn.commit()

    def search(self, query: str) -> list[dict]:
        if not self._conn:
            return []
        like = f"%{query.lower()}%"
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT uid, kind, key, value, metadata::text, created_at::text
                FROM memory_items
                WHERE LOWER(kind) LIKE %s OR LOWER(key) LIKE %s OR LOWER(value) LIKE %s
                ORDER BY created_at DESC
                LIMIT 100
                """,
                (like, like, like),
            )
            rows = cur.fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "uid": r[0],
                    "kind": r[1],
                    "key": r[2],
                    "value": r[3],
                    "metadata": json.loads(r[4] or "{}"),
                    "created_at": r[5],
                    "source": "postgres",
                }
            )
        return out
