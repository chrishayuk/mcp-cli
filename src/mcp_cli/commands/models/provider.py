# src/mcp_cli/commands/models/provider.py
"""Provider command models."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ProviderActionParams(BaseModel):
    """Parameters for provider actions."""

    args: List[str] = Field(default_factory=list, description="Command arguments")
    detailed: bool = Field(default=False, description="Show detailed information")

    model_config = {"frozen": False}


class ProviderInfo(BaseModel):
    """Information about a provider."""

    name: str = Field(description="Provider name")
    is_current: bool = Field(default=False, description="Is this the current provider")
    is_available: bool = Field(default=True, description="Is this provider available")
    requires_api_key: bool = Field(
        default=True, description="Does this provider require an API key"
    )
    api_key_configured: bool = Field(default=False, description="Is API key configured")
    description: Optional[str] = Field(default=None, description="Provider description")
    models: List[str] = Field(default_factory=list, description="Available models")

    model_config = {"frozen": False}
