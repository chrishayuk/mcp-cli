# src/mcp_cli/commands/models/responses.py
"""Response models for command actions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import Field, field_validator

from .base_model import CommandBaseModel


class ServerInfoResponse(CommandBaseModel):
    """
    Response model for server information.

    Example:
        >>> server = ServerInfoResponse(
        ...     name="my-server",
        ...     transport="stdio",
        ...     capabilities={"tools": True},
        ...     tool_count=5,
        ...     status="connected",
        ...     ping_ms=25.5
        ... )
    """

    name: str = Field(min_length=1, description="Server name")
    transport: str = Field(pattern="^(stdio|http|sse)$", description="Transport type")
    capabilities: Dict[str, Any] = Field(
        default_factory=dict, description="Server capabilities"
    )
    tool_count: int = Field(default=0, ge=0, description="Number of tools")
    status: str = Field(min_length=1, description="Server status")
    ping_ms: Optional[float] = Field(
        default=None, ge=0, description="Ping latency in milliseconds"
    )

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v: str) -> str:
        """Validate transport type."""
        if v not in ("stdio", "http", "sse"):
            raise ValueError(f"Invalid transport: {v}. Must be stdio, http, or sse")
        return v


class ResourceInfoResponse(CommandBaseModel):
    """
    Response model for resource information.

    Example:
        >>> resource = ResourceInfoResponse(
        ...     uri="file:///path/to/file.txt",
        ...     name="file.txt",
        ...     mime_type="text/plain",
        ...     server="my-server"
        ... )
    """

    uri: str = Field(min_length=1, description="Resource URI")
    name: Optional[str] = Field(default=None, description="Resource name")
    description: Optional[str] = Field(default=None, description="Resource description")
    mime_type: Optional[str] = Field(default=None, description="MIME type")
    server: str = Field(min_length=1, description="Server name providing this resource")


class PromptInfoResponse(CommandBaseModel):
    """
    Response model for prompt information.

    Example:
        >>> prompt = PromptInfoResponse(
        ...     name="generate-code",
        ...     description="Generate code from description",
        ...     arguments=[{"name": "language", "required": True}],
        ...     server="my-server"
        ... )
    """

    name: str = Field(min_length=1, description="Prompt name")
    description: Optional[str] = Field(default=None, description="Prompt description")
    arguments: List[Dict[str, Any]] = Field(
        default_factory=list, description="Prompt arguments"
    )
    server: str = Field(min_length=1, description="Server name providing this prompt")


class ToolInfoResponse(CommandBaseModel):
    """
    Response model for tool information.

    Example:
        >>> tool = ToolInfoResponse(
        ...     name="search",
        ...     namespace="my-server",
        ...     description="Search for files",
        ...     parameters={"query": {"type": "string"}}
        ... )
    """

    name: str = Field(min_length=1, description="Tool name")
    namespace: str = Field(min_length=1, description="Tool namespace/server")
    description: Optional[str] = Field(default=None, description="Tool description")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters schema"
    )
