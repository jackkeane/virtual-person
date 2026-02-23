from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import json


@dataclass
class AuditEvent:
    type: str
    detail: str
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class AuditLogger:
    def __init__(self, persist_path: str | None = None) -> None:
        self.events: list[AuditEvent] = []
        self._persist_path = Path(persist_path) if persist_path else None

    def log(self, event_type: str, detail: str) -> None:
        event = AuditEvent(type=event_type, detail=detail)
        self.events.append(event)
        if self._persist_path:
            self._append_to_file(event)

    def _append_to_file(self, event: AuditEvent) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._persist_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def list_events(self) -> list[dict]:
        return [asdict(e) for e in self.events]
