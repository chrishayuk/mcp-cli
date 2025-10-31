# src/mcp_cli/commands/models/model.py
"""Model command models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelActionParams(BaseModel):
    """Parameters for model actions."""

    args: List[str] = Field(default_factory=list, description="Command arguments")
    provider: Optional[str] = Field(default=None, description="Provider name")
    detailed: bool = Field(default=False, description="Show detailed information")

    model_config = {"frozen": False}


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str = Field(description="Model name")
    provider: str = Field(description="Provider name")
    is_current: bool = Field(default=False, description="Is this the current model")
    description: Optional[str] = Field(default=None, description="Model description")
    capabilities: Dict[str, Any] = Field(
        default_factory=dict, description="Model capabilities"
    )

    model_config = {"frozen": False}
