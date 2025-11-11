# src/mcp_cli/commands/models/tool.py
"""Tool command models."""

from __future__ import annotations

from typing import Any
from pydantic import Field

from mcp_cli.commands.models.base_model import CommandBaseModel


class ToolActionParams(CommandBaseModel):
    """Parameters for tool actions."""

    args: list[str] = Field(default_factory=list, description="Command arguments")
    detailed: bool = Field(default=False, description="Show detailed information")
    namespace: str | None = Field(default=None, description="Filter by namespace")


class ToolCallParams(CommandBaseModel):
    """Parameters for calling a tool."""

    tool_name: str = Field(description="Fully qualified tool name")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )
    confirm: bool = Field(default=True, description="Confirm before execution")
