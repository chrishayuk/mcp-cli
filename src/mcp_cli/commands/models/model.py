# src/mcp_cli/commands/models/model.py
"""Model command models."""

from __future__ import annotations

from typing import Any
from pydantic import Field

from mcp_cli.commands.models.base_model import CommandBaseModel


class ModelActionParams(CommandBaseModel):
    """Parameters for model actions."""

    args: list[str] = Field(default_factory=list, description="Command arguments")
    provider: str | None = Field(default=None, description="Provider name")
    detailed: bool = Field(default=False, description="Show detailed information")


class ModelInfo(CommandBaseModel):
    """Information about an available model."""

    name: str = Field(min_length=1, description="Model name")
    provider: str = Field(min_length=1, description="Provider name")
    is_current: bool = Field(default=False, description="Is this the current model")
    description: str | None = Field(default=None, description="Model description")
    capabilities: dict[str, Any] = Field(
        default_factory=dict, description="Model capabilities"
    )
