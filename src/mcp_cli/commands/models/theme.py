# src/mcp_cli/commands/models/theme.py
"""Theme command models."""

from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel, Field


class ThemeActionParams(BaseModel):
    """Parameters for theme actions."""

    theme_name: Optional[str] = Field(default=None, description="Theme name to set")
    list_themes: bool = Field(default=False, description="List available themes")

    model_config = {"frozen": False}


class ThemeInfo(BaseModel):
    """Information about a theme."""

    name: str = Field(description="Theme name")
    is_current: bool = Field(default=False, description="Is this the current theme")
    description: Optional[str] = Field(default=None, description="Theme description")
    colors: Dict[str, str] = Field(
        default_factory=dict, description="Theme color palette"
    )

    model_config = {"frozen": False}
