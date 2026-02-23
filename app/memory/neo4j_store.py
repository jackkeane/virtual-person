from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass
class Neo4jConfig:
    uri: str
    username: str
    password: str


class Neo4jMemoryStore:
    def __init__(self, cfg: Neo4jConfig) -> None:
        self.cfg = cfg
        self._driver = None

    def connect(self) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(self.cfg.uri, auth=(self.cfg.username, self.cfg.password))
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT memory_item_key IF NOT EXISTS FOR (m:MemoryItem) REQUIRE m.uid IS UNIQUE")

    def available(self) -> bool:
        return self._driver is not None

    def write(self, kind: str, key: str, value: str, uid: str, metadata: dict | None = None) -> None:
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                """
                MERGE (m:MemoryItem {uid: $uid})
                SET m.kind = $kind,
                    m.key = $key,
                    m.value = $value,
                    m.metadata = $metadata,
                    m.created_at = datetime()
                """,
                uid=uid,
                kind=kind,
                key=key,
                value=value,
                metadata=json.dumps(metadata or {}, ensure_ascii=False),
            )

    def delete(self, kind: str, key: str) -> None:
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                "MATCH (m:MemoryItem) WHERE m.kind = $kind AND m.key = $key DETACH DELETE m",
                kind=kind,
                key=key,
            )

    def delete_all(self) -> None:
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run("MATCH (m:MemoryItem) DETACH DELETE m")

    def search(self, query: str) -> list[dict]:
        if not self._driver:
            return []
        q = query.lower()
        with self._driver.session() as session:
            res = session.run(
                """
                MATCH (m:MemoryItem)
                WHERE toLower(m.kind) CONTAINS $q OR toLower(m.key) CONTAINS $q OR toLower(m.value) CONTAINS $q
                RETURN m.uid AS uid, m.kind AS kind, m.key AS key, m.value AS value,
                       m.metadata AS metadata, toString(m.created_at) AS created_at
                ORDER BY m.created_at DESC
                LIMIT 100
                """,
                q=q,
            )
            rows = [r.data() for r in res]
        for r in rows:
            try:
                r["metadata"] = json.loads(r.get("metadata") or "{}")
            except Exception:
                r["metadata"] = {}
            r["source"] = "neo4j"
        return rows
