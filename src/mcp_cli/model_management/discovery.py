# src/mcp_cli/models/discovery.py
"""
Model discovery results.

Pydantic models for representing the results of model discovery operations.
"""

from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field, model_validator


class DiscoveryResult(BaseModel):
    """
    Result of model discovery from an API.

    This model represents the outcome of attempting to discover available
    models from a provider's API endpoint.
    """

    provider: str = Field(..., description="Provider name")
    models: List[str] = Field(default_factory=list, description="Discovered models")
    success: bool = Field(..., description="Whether discovery succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")
    discovered_count: int = Field(default=0, description="Number of models discovered")

    @model_validator(mode="after")
    def count_models_after(self):
        """Auto-calculate discovered_count from models list if not explicitly provided."""
        # If discovered_count is 0 and models exist, auto-calculate
        if self.discovered_count == 0 and self.models:
            object.__setattr__(self, "discovered_count", len(self.models))
        return self

    @property
    def has_models(self) -> bool:
        """Check if any models were discovered."""
        return len(self.models) > 0

    @property
    def error_message(self) -> str:
        """Get a formatted error message."""
        if self.error:
            return f"Discovery failed for {self.provider}: {self.error}"
        return ""

    model_config = {"frozen": True}  # Immutable result
