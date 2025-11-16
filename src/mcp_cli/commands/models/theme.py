# src/mcp_cli/commands/models/theme.py
"""Theme command models."""

from __future__ import annotations

from pydantic import Field

from mcp_cli.commands.models.base_model import CommandBaseModel


class ThemeActionParams(CommandBaseModel):
    """Parameters for theme actions."""

    theme_name: str | None = Field(default=None, description="Theme name to set")
    list_themes: bool = Field(default=False, description="List available themes")


class ThemeInfo(CommandBaseModel):
    """Information about a theme."""

    name: str = Field(description="Theme name")
    is_current: bool = Field(default=False, description="Is this the current theme")
    description: str | None = Field(default=None, description="Theme description")
    colors: dict[str, str] = Field(
        default_factory=dict, description="Theme color palette"
    )
