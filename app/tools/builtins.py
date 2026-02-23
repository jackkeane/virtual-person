"""
Built-in tools for Phase 2.
"""
from __future__ import annotations

from datetime import datetime, UTC

from app.tools.registry import RiskTier, ToolRegistry


def register_builtins(registry: ToolRegistry, memory_svc=None, proactivity_svc=None) -> None:
    """Register all built-in tools."""

    # --- Read-tier tools ---

    def tool_search_memory(query: str = "") -> dict:
        if not memory_svc:
            return {"items": []}
        items = memory_svc.search(query)
        return {"items": [{"kind": i.kind, "key": i.key, "value": i.value} for i in items[:10]]}

    registry.register(
        name="search_memory",
        description="Search stored memories by keyword",
        risk=RiskTier.READ,
        handler=tool_search_memory,
    )

    def tool_get_time() -> dict:
        now = datetime.now(UTC)
        return {"utc": now.isoformat(), "readable": now.strftime("%Y-%m-%d %H:%M UTC")}

    registry.register(
        name="get_time",
        description="Get current UTC time",
        risk=RiskTier.READ,
        handler=tool_get_time,
    )

    # --- Write-tier tools ---

    def tool_save_memory(kind: str = "note", key: str = "", value: str = "") -> dict:
        if not memory_svc:
            return {"saved": False}
        item = memory_svc.write(kind, key, value)
        return {"saved": True, "kind": kind, "key": key}

    registry.register(
        name="save_memory",
        description="Save a fact or note to memory",
        risk=RiskTier.WRITE,
        handler=tool_save_memory,
    )

    def tool_set_reminder(message: str = "", minutes: int = 30) -> dict:
        if not proactivity_svc:
            return {"set": False}
        remind_at = datetime.now(UTC).isoformat()
        proactivity_svc.add_reminder(message=message, minutes=minutes)
        return {"set": True, "message": message, "in_minutes": minutes}

    registry.register(
        name="set_reminder",
        description="Set a reminder for later",
        risk=RiskTier.WRITE,
        handler=tool_set_reminder,
    )

    # --- External-tier tools (need confirmation) ---

    def tool_send_message(to: str = "", text: str = "") -> dict:
        # Stub — would integrate with messaging adapter
        return {"sent": True, "to": to, "preview": text[:80]}

    registry.register(
        name="send_message",
        description="Send a message to someone (requires confirmation)",
        risk=RiskTier.EXTERNAL,
        handler=tool_send_message,
    )

    def tool_web_search(query: str = "") -> dict:
        # Stub — would integrate with search adapter
        return {"results": [f"[stub result for '{query}']"]}

    registry.register(
        name="web_search",
        description="Search the web (requires confirmation)",
        risk=RiskTier.EXTERNAL,
        handler=tool_web_search,
    )
