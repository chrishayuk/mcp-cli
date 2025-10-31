# src/mcp_cli/commands/models/token.py
"""Token command models."""

from __future__ import annotations

from typing import List, Optional
from pydantic import Field

from .base_model import CommandBaseModel


class TokenListParams(CommandBaseModel):
    """Parameters for token list action."""

    namespace: Optional[str] = Field(default=None, description="Filter by namespace")
    show_oauth: bool = Field(default=True, description="Show OAuth tokens")
    show_bearer: bool = Field(default=True, description="Show bearer tokens")
    show_api_keys: bool = Field(default=True, description="Show API keys")
    show_providers: bool = Field(default=True, description="Show provider API keys")
    server_names: List[str] = Field(
        default_factory=list,
        description="List of configured servers to check for OAuth tokens",
    )
    provider_names: List[str] = Field(
        default_factory=list,
        description="List of configured providers to check for tokens",
    )


class TokenSetParams(CommandBaseModel):
    """Parameters for token set action."""

    name: str = Field(description="Token identifier/name")
    token_type: str = Field(
        default="bearer", description="Token type (bearer/api-key/generic)"
    )
    value: Optional[str] = Field(
        default=None, description="Token value (prompts if not provided)"
    )
    provider: Optional[str] = Field(
        default=None, description="Provider name (for API keys)"
    )
    namespace: str = Field(default="generic", description="Storage namespace")


class TokenDeleteParams(CommandBaseModel):
    """Parameters for token delete action."""

    name: str = Field(description="Token identifier/name")
    namespace: Optional[str] = Field(default=None, description="Storage namespace")
    oauth: bool = Field(default=False, description="Delete OAuth token for server")


class TokenClearParams(CommandBaseModel):
    """Parameters for token clear action."""

    namespace: Optional[str] = Field(
        default=None, description="Clear only this namespace"
    )
    force: bool = Field(default=False, description="Skip confirmation prompt")


class TokenProviderParams(CommandBaseModel):
    """Parameters for provider token operations."""

    provider: str = Field(description="Provider name")
    api_key: Optional[str] = Field(default=None, description="API key value")
