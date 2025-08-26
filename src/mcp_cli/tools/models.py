# mcp_cli/tools/models.py
"""Data models used throughout MCP-CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Tool-related models (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ToolInfo:
    """Information about a tool."""

    name: str
    namespace: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    is_async: bool = False
    tags: List[str] = field(default_factory=list)
    supports_streaming: bool = False  # Add this field


@dataclass
class ServerInfo:
    """Information about a connected server instance."""

    id: int
    name: str
    status: str
    tool_count: int
    namespace: str


@dataclass
class ToolCallResult:
    """Outcome of a tool execution."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────────────
# NEW - resource-related models
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ResourceInfo:
    """
    Canonical representation of *one* resource entry as returned by
    ``resources.list``.

    The MCP spec does not prescribe a single shape, so we normalise the common
    fields we use in the UI.  **All additional keys** are preserved inside
    ``extra``.
    """

    # Common attributes we frequently need in the UI
    id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None

    # Anything else goes here …
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Factory helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_raw(cls, raw: Any) -> "ResourceInfo":
        """
        Convert a raw list item (dict | str | int | …) into a ResourceInfo.

        If *raw* is not a mapping we treat it as an opaque scalar and store it
        in ``extra["value"]`` so it is never lost.
        """
        if isinstance(raw, dict):
            known = {k: raw.get(k) for k in ("id", "name", "type")}
            extra = {k: v for k, v in raw.items() if k not in known}
            return cls(**known, extra=extra)
        # primitive - wrap it
        return cls(extra={"value": raw})
