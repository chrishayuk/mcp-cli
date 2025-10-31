# src/mcp_cli/commands/models/resource.py
"""Resource command models."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ResourceActionParams(BaseModel):
    """Parameters for resource actions."""

    args: List[str] = Field(default_factory=list, description="Command arguments")
    detailed: bool = Field(default=False, description="Show detailed information")
    server: Optional[str] = Field(default=None, description="Filter by server")

    model_config = {"frozen": False}
