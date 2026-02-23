"""Memory curation: importance scoring, dedup, staleness detection, relevance ranking."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re


@dataclass
class ScoredMemory:
    kind: str
    key: str
    value: str
    created_at: str
    score: float
    source: str = "in-memory"


KIND_WEIGHTS: dict[str, float] = {
    "identity": 1.0,
    "preference": 0.8,
    "project": 0.7,
    "goal": 0.9,
    "todo": 0.6,
    "note": 0.4,
    "ephemeral": 0.2,
}

KIND_TTL_DAYS: dict[str, int | None] = {
    "identity": None,
    "preference": None,
    "project": 180,
    "goal": 365,
    "todo": 30,
    "note": 90,
    "ephemeral": 7,
}


def importance_score(kind: str, keyword_overlap: float = 0.0) -> float:
    base = KIND_WEIGHTS.get(kind, 0.3)
    return round(min(base + keyword_overlap * 0.35, 1.0), 3)


def relevance_overlap(query: str, key: str, value: str) -> float:
    tokens = [t for t in re.split(r"\W+", query.lower()) if len(t) >= 2]
    if not tokens:
        return 0.0
    hay = f"{key} {value}".lower()
    matched = sum(1 for t in tokens if t in hay)
    return matched / len(tokens)


def _recency_bonus(created_at: str) -> float:
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        age_days = (datetime.now(UTC) - dt).total_seconds() / 86400
        if age_days < 1:
            return 0.12
        if age_days < 7:
            return 0.08
        if age_days < 30:
            return 0.04
    except Exception:
        pass
    return 0.0


def retrieval_score(kind: str, key: str, value: str, created_at: str, query: str) -> float:
    overlap = relevance_overlap(query, key, value)
    exact_key = 0.2 if query.strip().lower() == key.strip().lower() else 0.0
    return round(min(importance_score(kind, overlap) + _recency_bonus(created_at) + exact_key, 1.5), 3)


def is_stale(kind: str, created_at: str) -> bool:
    ttl_days = KIND_TTL_DAYS.get(kind)
    if ttl_days is None:
        return False
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        age = datetime.now(UTC) - created
        return age > timedelta(days=ttl_days)
    except Exception:
        return False


def dedup_memories(items: list[ScoredMemory]) -> list[ScoredMemory]:
    seen: dict[tuple[str, str, str], ScoredMemory] = {}
    for item in items:
        k = (item.kind, item.key, item.value)
        if k not in seen or item.score > seen[k].score:
            seen[k] = item
    return list(seen.values())


def rank_memories(items: list[ScoredMemory], limit: int = 10) -> list[ScoredMemory]:
    fresh = [i for i in items if not is_stale(i.kind, i.created_at)]
    deduped = dedup_memories(fresh)
    deduped.sort(key=lambda x: x.score, reverse=True)
    return deduped[:limit]
