"""Pydantic models for playbook command parameters."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlaybookActionParams(BaseModel):
    """Parameters for playbook actions (management only, not querying)."""

    action: str = Field(default="status", description="Action: enable, disable, status, list")
    args: list[str] = Field(default_factory=list, description="Additional arguments")

    model_config = {"frozen": False}
