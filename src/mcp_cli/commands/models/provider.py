# src/mcp_cli/commands/models/provider.py
"""Provider command models."""

from __future__ import annotations

from pydantic import Field

from mcp_cli.commands.models.base_model import CommandBaseModel


class ProviderActionParams(CommandBaseModel):
    """Parameters for provider actions."""

    args: list[str] = Field(default_factory=list, description="Command arguments")
    detailed: bool = Field(default=False, description="Show detailed information")


class ProviderInfo(CommandBaseModel):
    """Information about a provider."""

    name: str = Field(min_length=1, description="Provider name")
    is_current: bool = Field(default=False, description="Is this the current provider")
    is_available: bool = Field(default=True, description="Is this provider available")
    requires_api_key: bool = Field(
        default=True, description="Does this provider require an API key"
    )
    api_key_configured: bool = Field(default=False, description="Is API key configured")
    description: str | None = Field(default=None, description="Provider description")
    models: list[str] = Field(default_factory=list, description="Available models")
