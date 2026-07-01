"""
Session service: per-user conversation history for multi-turn chat.

Backend selection mirrors app/memory/service.py:_init_postgres -- when
``config.session_backend`` selects redis AND Redis is reachable, history is
delegated to :class:`app.session.redis_store.RedisSessionStore`; otherwise the
in-memory store below is used unchanged. With REDIS_URL unset (the default and
the test default) ``_redis`` is ``None`` and behavior is identical to before
Feature 3.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from app.config import config
from app.infra.redis_client import redis_available
from app.session.redis_store import RedisSessionStore


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class SessionService:
    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._history: dict[str, list[Turn]] = defaultdict(list)
        self._redis: RedisSessionStore | None = self._init_redis()

    def _init_redis(self) -> RedisSessionStore | None:
        """Build a Redis-backed store, or None to keep the in-memory fallback.

        Gated exactly like _init_postgres: select redis only when configured
        (``session_backend`` in {"auto", "redis"}) AND the backend pings OK.
        Any failure -> None -> in-memory.
        """
        if config.session_backend not in ("auto", "redis"):
            return None
        try:
            if not redis_available():
                return None
            return RedisSessionStore(self.max_turns)
        except Exception:
            return None

    def add(self, user_id: str, role: str, content: str) -> None:
        if self._redis is not None:
            self._redis.add(user_id, role, content)
            return
        turns = self._history[user_id]
        turns.append(Turn(role=role, content=content))
        if len(turns) > self.max_turns:
            self._history[user_id] = turns[-self.max_turns:]

    def get(self, user_id: str, limit: int | None = None) -> list[dict]:
        if self._redis is not None:
            return self._redis.get(user_id, limit=limit)
        turns = self._history[user_id]
        if limit:
            turns = turns[-limit:]
        return [{"role": t.role, "content": t.content} for t in turns]

    def clear(self, user_id: str) -> None:
        if self._redis is not None:
            self._redis.clear(user_id)
            return
        self._history.pop(user_id, None)
