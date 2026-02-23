from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

AvatarState = Literal["idle", "listening", "thinking", "speaking"]


@dataclass
class StateSnapshot:
    state: AvatarState
    since: str
    expression: dict


class ConversationStateMachine:
    """Simple deterministic state machine for avatar conversation flow."""

    _allowed: dict[AvatarState, set[AvatarState]] = {
        "idle": {"listening", "thinking", "speaking"},
        "listening": {"thinking", "idle"},
        "thinking": {"speaking", "idle"},
        "speaking": {"idle", "thinking", "listening"},
    }

    _presets: dict[AvatarState, dict] = {
        "idle": {"eyes": 0.7, "brows": 0.5, "mouth": 0.2, "head": 0.0},
        "listening": {"eyes": 0.85, "brows": 0.6, "mouth": 0.1, "head": 0.1},
        "thinking": {"eyes": 0.55, "brows": 0.45, "mouth": 0.15, "head": 0.2},
        "speaking": {"eyes": 0.75, "brows": 0.55, "mouth": 0.55, "head": 0.05},
    }

    def __init__(self) -> None:
        self._state: AvatarState = "idle"
        self._since = datetime.now(UTC)

    @property
    def state(self) -> AvatarState:
        return self._state

    def transition(self, new_state: AvatarState) -> StateSnapshot:
        # Any state is always allowed to return to idle.
        allowed_targets = set(self._allowed[self._state])
        allowed_targets.add("idle")

        if new_state != self._state and new_state not in allowed_targets:
            print(f"[StateMachine] Warning: illegal transition {self._state} -> {new_state}; keeping current state")
            return self.snapshot()

        self._state = new_state
        self._since = datetime.now(UTC)
        return self.snapshot()

    def auto_to_idle(self, timeout_ms: int = 10_000) -> StateSnapshot:
        elapsed = int((datetime.now(UTC) - self._since).total_seconds() * 1000)
        if self._state in {"thinking", "speaking"} and elapsed >= timeout_ms:
            return self.transition("idle")
        return self.snapshot()

    def expression_for(self, state: AvatarState | None = None) -> dict:
        s = state or self._state
        return dict(self._presets[s])

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            state=self._state,
            since=self._since.isoformat(),
            expression=self.expression_for(),
        )
