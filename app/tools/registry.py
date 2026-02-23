"""
Tool registry with risk-tiered permission model.

Risk tiers:
  - read:     no confirmation needed (e.g. search memory, check weather)
  - write:    logged, auto-approved within session (e.g. save note, set reminder)
  - external: requires explicit user confirmation (e.g. send email, post message)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable


class RiskTier(str, Enum):
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"


@dataclass
class ToolDef:
    name: str
    description: str
    risk: RiskTier
    handler: Callable[..., Any] | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._pending_confirmations: dict[str, dict] = {}

    def register(self, name: str, description: str, risk: RiskTier, handler: Callable[..., Any] | None = None) -> None:
        self._tools[name] = ToolDef(name=name, description=description, risk=risk, handler=handler)

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "risk": t.risk.value}
            for t in self._tools.values()
        ]

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def execute(self, name: str, params: dict | None = None, user_confirmed: bool = False) -> dict:
        tool = self._tools.get(name)
        if not tool:
            return {"ok": False, "error": f"Unknown tool: {name}"}

        params = params or {}

        # External tools need confirmation
        if tool.risk == RiskTier.EXTERNAL and not user_confirmed:
            confirm_id = f"confirm_{name}_{id(params)}"
            self._pending_confirmations[confirm_id] = {"tool": name, "params": params}
            return {
                "ok": False,
                "needs_confirmation": True,
                "confirm_id": confirm_id,
                "message": f"Action '{name}' requires your confirmation before executing. Reply with confirm_id to approve.",
            }

        if tool.handler:
            try:
                result = tool.handler(**params)
                return {"ok": True, "tool": name, "risk": tool.risk.value, "result": result}
            except Exception as e:
                return {"ok": False, "tool": name, "error": str(e)}

        return {"ok": True, "tool": name, "risk": tool.risk.value, "result": f"[stub] {name} executed"}

    def confirm(self, confirm_id: str) -> dict:
        pending = self._pending_confirmations.pop(confirm_id, None)
        if not pending:
            return {"ok": False, "error": "No pending confirmation with that id"}
        return self.execute(pending["tool"], pending["params"], user_confirmed=True)
