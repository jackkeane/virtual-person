"""
Proactivity service with reminder CRUD, daily summaries, and anti-annoyance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
import uuid


@dataclass
class Reminder:
    id: str
    message: str
    due_at: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    fired: bool = False
    reason: str = "user-set"


class ProactivityService:
    def __init__(self, quiet_start: int, quiet_end: int, cooldown_minutes: int) -> None:
        self.quiet_start = quiet_start
        self.quiet_end = quiet_end
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self._last_sent_at: datetime | None = None
        self._reminders: list[Reminder] = []

    # --- Anti-annoyance ---

    def is_quiet_hours(self, now: datetime) -> bool:
        h = now.hour
        if self.quiet_start > self.quiet_end:
            return h >= self.quiet_start or h < self.quiet_end
        return self.quiet_start <= h < self.quiet_end

    def can_send(self, now: datetime) -> tuple[bool, str]:
        if self.is_quiet_hours(now):
            return False, "quiet-hours"
        if self._last_sent_at and now - self._last_sent_at < self.cooldown:
            return False, "cooldown"
        return True, "ok"

    def mark_sent(self, now: datetime) -> None:
        self._last_sent_at = now

    # --- Reminder CRUD ---

    def add_reminder(self, message: str, minutes: int = 30, reason: str = "user-set") -> Reminder:
        r = Reminder(
            id=str(uuid.uuid4())[:8],
            message=message,
            due_at=datetime.now(UTC) + timedelta(minutes=minutes),
            reason=reason,
        )
        self._reminders.append(r)
        return r

    def list_reminders(self, include_fired: bool = False) -> list[dict]:
        items = self._reminders if include_fired else [r for r in self._reminders if not r.fired]
        return [
            {
                "id": r.id,
                "message": r.message,
                "due_at": r.due_at.isoformat(),
                "fired": r.fired,
                "reason": r.reason,
            }
            for r in items
        ]

    def cancel_reminder(self, reminder_id: str) -> bool:
        for r in self._reminders:
            if r.id == reminder_id and not r.fired:
                self._reminders.remove(r)
                return True
        return False

    def check_due(self, now: datetime | None = None) -> list[Reminder]:
        """Return and mark-fired all reminders that are due."""
        now = now or datetime.now(UTC)
        due = []
        for r in self._reminders:
            if not r.fired and r.due_at <= now:
                r.fired = True
                due.append(r)
        return due

    # --- Daily summary ---

    def daily_summary(self, memory_items: list[dict] | None = None) -> dict:
        """Generate a lightweight daily summary payload."""
        now = datetime.now(UTC)
        upcoming = [
            {"id": r.id, "message": r.message, "due_at": r.due_at.isoformat()}
            for r in self._reminders
            if not r.fired and r.due_at <= now + timedelta(hours=24)
        ]
        overdue = [
            {"id": r.id, "message": r.message, "due_at": r.due_at.isoformat()}
            for r in self._reminders
            if not r.fired and r.due_at < now
        ]
        return {
            "generated_at": now.isoformat(),
            "upcoming_reminders_24h": upcoming,
            "overdue_reminders": overdue,
            "active_reminder_count": sum(1 for r in self._reminders if not r.fired),
            "recent_memory_count": len(memory_items) if memory_items else 0,
        }
