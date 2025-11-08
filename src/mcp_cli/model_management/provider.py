# src/mcp_cli/models/provider.py
"""
Provider configuration models.

Pydantic models for managing provider configurations in a type-safe manner.
"""

from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field, model_validator


class RuntimeProviderConfig(BaseModel):
    """
    Configuration for a runtime provider (OpenAI-compatible API).

    This model represents a provider that can be added at runtime or loaded
    from preferences, with full type safety and validation.
    """

    name: str = Field(..., description="Provider name")
    api_base: str = Field(..., description="API base URL")
    models: List[str] = Field(default_factory=list, description="Available models")
    default_model: Optional[str] = Field(
        None, description="Default model (first in models list if not specified)"
    )
    api_key: Optional[str] = Field(
        None, description="API key (stored in memory only for runtime providers)"
    )
    is_runtime: bool = Field(
        default=True, description="True if added at runtime (not persisted)"
    )

    @model_validator(mode="after")
    def set_default_model_if_needed(self):
        """Set default_model to first model if not specified."""
        if self.default_model is None and self.models:
            self.default_model = self.models[0]
        return self

    @property
    def has_models(self) -> bool:
        """Check if provider has any models configured."""
        return len(self.models) > 0

    def add_models(self, models: List[str]) -> None:
        """
        Add models to the provider and update default if needed.

        Args:
            models: List of model IDs to add
        """
        self.models.extend(models)
        if not self.default_model and self.models:
            self.default_model = self.models[0]

    def set_models(self, models: List[str]) -> None:
        """
        Replace all models and update default.

        Args:
            models: New list of model IDs
        """
        self.models = models
        if models and not self.default_model:
            self.default_model = models[0]

    model_config = {"frozen": False}  # Allow mutation for updates


class ProviderCapabilities(BaseModel):
    """
    Capabilities and features supported by a provider/model.

    This model describes what features a specific provider or model supports,
    such as streaming, function calling, vision, etc.
    """

    supports_streaming: bool = Field(default=False, description="Streaming support")
    supports_tools: bool = Field(default=False, description="Function calling support")
    supports_vision: bool = Field(default=False, description="Vision/image support")
    supports_json_mode: bool = Field(default=False, description="JSON mode support")
    max_context_length: Optional[int] = Field(
        None, description="Maximum context length in tokens"
    )
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens")

    model_config = {"frozen": True}  # Immutable
