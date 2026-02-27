from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MCPToolResult:
    """Normalized payload returned by local MCP tool adapters."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    tables: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MCPToolResult":
        return cls(
            success=bool(payload.get("success", False)),
            data=payload.get("data") or {},
            artifacts=payload.get("artifacts") or [],
            tables=payload.get("tables") or {},
            messages=payload.get("messages") or [],
            error=payload.get("error"),
        )

