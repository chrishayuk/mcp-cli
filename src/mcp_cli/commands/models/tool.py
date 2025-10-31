# src/mcp_cli/commands/models/tool.py
"""Tool command models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolActionParams(BaseModel):
    """Parameters for tool actions."""

    args: List[str] = Field(default_factory=list, description="Command arguments")
    detailed: bool = Field(default=False, description="Show detailed information")
    namespace: Optional[str] = Field(default=None, description="Filter by namespace")

    model_config = {"frozen": False}


class ToolCallParams(BaseModel):
    """Parameters for calling a tool."""

    tool_name: str = Field(description="Fully qualified tool name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )
    confirm: bool = Field(default=True, description="Confirm before execution")

    model_config = {"frozen": False}
