"""
Session service: per-user conversation history for multi-turn chat.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class SessionService:
    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._history: dict[str, list[Turn]] = defaultdict(list)

    def add(self, user_id: str, role: str, content: str) -> None:
        turns = self._history[user_id]
        turns.append(Turn(role=role, content=content))
        if len(turns) > self.max_turns:
            self._history[user_id] = turns[-self.max_turns:]

    def get(self, user_id: str, limit: int | None = None) -> list[dict]:
        turns = self._history[user_id]
        if limit:
            turns = turns[-limit:]
        return [{"role": t.role, "content": t.content} for t in turns]

    def clear(self, user_id: str) -> None:
        self._history.pop(user_id, None)
