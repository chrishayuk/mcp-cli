# src/mcp_cli/commands/models/server.py
"""Server command models."""

from __future__ import annotations

from typing import Any, cast

from pydantic import Field, field_validator

from .base_model import CommandBaseModel


class ServerActionParams(CommandBaseModel):
    """
    Parameters for server actions.

    Examples:
        List all servers:
        >>> params = ServerActionParams()

        List with details and ping:
        >>> params = ServerActionParams(detailed=True, ping_servers=True)

        JSON output format:
        >>> params = ServerActionParams(output_format="json")
    """

    args: list[str] = Field(default_factory=list, description="Command arguments")
    detailed: bool = Field(default=False, description="Show detailed information")
    show_capabilities: bool = Field(
        default=False, description="Show server capabilities"
    )
    show_transport: bool = Field(default=False, description="Show transport details")
    output_format: str = Field(
        default="table",
        description="Output format (table/json)",
        pattern="^(table|json)$",
    )
    ping_servers: bool = Field(default=False, description="Test server connectivity")

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format is supported."""
        if v not in ("table", "json"):
            raise ValueError('output_format must be "table" or "json"')
        return v


class ServerStatusInfo(CommandBaseModel):
    """
    Status information for a server.

    Example:
        >>> status = ServerStatusInfo(
        ...     icon="✅",
        ...     status="Connected",
        ...     reason="Server is online and responding"
        ... )
    """

    icon: str = Field(min_length=1, description="Status icon")
    status: str = Field(min_length=1, description="Status text")
    reason: str = Field(min_length=1, description="Status reason")


class ServerPerformanceInfo(CommandBaseModel):
    """
    Performance information for a server.

    Example:
        >>> perf = ServerPerformanceInfo(
        ...     icon="🚀",
        ...     latency="25.5ms",
        ...     ping_ms=25.5
        ... )
    """

    icon: str = Field(min_length=1, description="Performance icon")
    latency: str = Field(min_length=1, description="Formatted latency")
    ping_ms: float | None = Field(
        default=None, ge=0, description="Raw ping in milliseconds"
    )

    @field_validator("ping_ms")
    @classmethod
    def validate_ping_ms(cls, v: float | None) -> float | None:
        """Validate ping time is non-negative."""
        if v is not None and v < 0:
            raise ValueError("ping_ms must be non-negative")
        return v


class ServerCapabilities(CommandBaseModel):
    """
    Server MCP capabilities.

    Example:
        >>> caps = ServerCapabilities(
        ...     tools=True,
        ...     prompts=True,
        ...     resources=False,
        ...     experimental={"events": True}
        ... )
        >>> caps.to_display_string()
        'Tools, Prompts, Events*'
    """

    tools: bool = Field(default=False, description="Supports tools")
    prompts: bool = Field(default=False, description="Supports prompts")
    resources: bool = Field(default=False, description="Supports resources")
    experimental: dict[str, Any] = Field(
        default_factory=dict, description="Experimental capabilities"
    )

    @property
    def has_events(self) -> bool:
        """Check if server has experimental events capability."""
        return cast(bool, self.experimental.get("events", False))

    @property
    def has_streaming(self) -> bool:
        """Check if server has experimental streaming capability."""
        return cast(bool, self.experimental.get("streaming", False))

    def to_display_string(self) -> str:
        """Format capabilities as readable string."""
        caps = []

        if self.tools:
            caps.append("Tools")
        if self.prompts:
            caps.append("Prompts")
        if self.resources:
            caps.append("Resources")

        if self.has_events:
            caps.append("Events*")
        if self.has_streaming:
            caps.append("Streaming*")

        return ", ".join(caps) if caps else "None"
