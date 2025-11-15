# src/mcp_cli/commands/models/provider.py
"""Provider command models."""

from __future__ import annotations

from enum import Enum
from pydantic import Field

from mcp_cli.commands.models.base_model import CommandBaseModel


class ProviderActionParams(CommandBaseModel):
    """Parameters for provider actions."""

    args: list[str] = Field(default_factory=list, description="Command arguments")
    detailed: bool = Field(default=False, description="Show detailed information")


class TokenSource(str, Enum):
    """Token source enumeration."""

    ENV = "env"
    STORAGE = "storage"
    NONE = "none"


class ProviderStatus(CommandBaseModel):
    """Provider status information."""

    icon: str = Field(description="Status icon (✅, ❌, ⚠️)")
    text: str = Field(description="Status text (Ready, Not Configured, etc.)")
    reason: str = Field(description="Detailed reason for the status")


class ProviderData(CommandBaseModel):
    """Complete provider data structure."""

    name: str = Field(description="Provider name")
    has_api_key: bool = Field(
        default=False, description="Whether API key is configured"
    )
    token_source: TokenSource = Field(
        default=TokenSource.NONE, description="Source of the API token"
    )
    models: list[str] = Field(default_factory=list, description="Available models")
    available_models: list[str] = Field(
        default_factory=list, description="Alternative key for available models"
    )
    default_model: str | None = Field(default=None, description="Default model")
    baseline_features: list[str] = Field(
        default_factory=list, description="Baseline features supported"
    )
    is_custom: bool = Field(default=False, description="Is this a custom provider")
    api_base: str | None = Field(default=None, description="API base URL")
    discovery_enabled: bool = Field(
        default=False, description="Is model discovery enabled"
    )
    error: str | None = Field(default=None, description="Error message if any")

    @property
    def all_models(self) -> list[str]:
        """Get all models from either models or available_models."""
        return self.models if self.models else self.available_models

    @property
    def model_count(self) -> int:
        """Get the count of available models."""
        return len(self.all_models)


class CustomProviderConfig(CommandBaseModel):
    """Configuration for a custom provider."""

    name: str = Field(description="Provider name")
    api_base: str = Field(description="API base URL")
    models: list[str] = Field(default_factory=list, description="Available models")
    default_model: str = Field(description="Default model")
    env_var_name: str | None = Field(
        default=None, description="Custom environment variable name for API key"
    )


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
