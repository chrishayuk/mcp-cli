# mcp_cli/tools/models.py
"""Data models used throughout MCP-CLI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Tool-related models (converted to Pydantic)
# ──────────────────────────────────────────────────────────────────────────────
class ToolInfo(BaseModel):
    """Information about a tool."""

    name: str
    namespace: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    is_async: bool = False
    tags: List[str] = Field(default_factory=list)
    supports_streaming: bool = False

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @property
    def fully_qualified_name(self) -> str:
        """Get the fully qualified tool name (namespace.name)."""
        return f"{self.namespace}.{self.name}" if self.namespace else self.name

    @property
    def display_name(self) -> str:
        """Get a user-friendly display name."""
        return self.name

    @property
    def has_parameters(self) -> bool:
        """Check if the tool has parameters defined."""
        return bool(self.parameters and self.parameters.get("properties"))

    @property
    def required_parameters(self) -> List[str]:
        """Get list of required parameter names."""
        if not self.parameters:
            return []
        required = self.parameters.get("required", [])
        return required if isinstance(required, list) else []

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "No description provided",
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


class ServerInfo(BaseModel):
    """Information about a connected server instance."""

    id: int
    name: str
    status: str
    tool_count: int
    namespace: str
    enabled: bool = True
    connected: bool = False
    transport: str = "stdio"  # "stdio", "http", "sse"
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None  # From server metadata
    version: Optional[str] = None  # Server version
    command: Optional[str] = None  # Server command if known
    args: List[str] = Field(default_factory=list)  # Command arguments
    env: Dict[str, str] = Field(default_factory=dict)  # Environment variables

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @property
    def is_healthy(self) -> bool:
        """Check if server is healthy and ready."""
        return self.status == "healthy" and self.connected

    @property
    def display_status(self) -> str:
        """Get a user-friendly status string."""
        if not self.enabled:
            return "disabled"
        elif not self.connected:
            return "disconnected"
        else:
            return self.status

    @property
    def display_description(self) -> str:
        """Get description or a default based on name."""
        # Use server-provided description if available
        if self.description:
            return self.description
        # Otherwise just return a generic description
        return f"{self.name} MCP server"

    @property
    def has_tools(self) -> bool:
        """Check if server has any tools."""
        return self.tool_count > 0


class ToolCallResult(BaseModel):
    """Outcome of a tool execution."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

    model_config = {"frozen": False, "arbitrary_types_allowed": True, "extra": "allow"}

    @property
    def display_result(self) -> str:
        """Get a display-friendly result string."""
        if not self.success:
            return f"Error: {self.error or 'Unknown error'}"
        elif isinstance(self.result, (dict, list)):
            import json

            return json.dumps(self.result, indent=2)
        else:
            return str(self.result)

    @property
    def has_error(self) -> bool:
        """Check if the result contains an error."""
        return not self.success or self.error is not None

    def to_conversation_history(self) -> str:
        """Format for inclusion in conversation history."""
        if self.success:
            return self.display_result
        else:
            return f"Tool execution failed: {self.error}"


# ──────────────────────────────────────────────────────────────────────────────
# NEW - resource-related models (converted to Pydantic)
# ──────────────────────────────────────────────────────────────────────────────
class ResourceInfo(BaseModel):
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
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

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
