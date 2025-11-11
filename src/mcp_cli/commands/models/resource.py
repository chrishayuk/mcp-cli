# src/mcp_cli/commands/models/resource.py
"""Resource command models."""

from __future__ import annotations

from pydantic import Field

from mcp_cli.commands.models.base_model import CommandBaseModel


class ResourceActionParams(CommandBaseModel):
    """Parameters for resource actions."""

    args: list[str] = Field(default_factory=list, description="Command arguments")
    detailed: bool = Field(default=False, description="Show detailed information")
    server: str | None = Field(default=None, description="Filter by server")
