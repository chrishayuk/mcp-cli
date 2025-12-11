"""Pydantic models for tool definitions - no more dict goop!"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolInputSchema(BaseModel):
    """Input schema for tool parameters (JSON Schema format)."""

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool = False

    model_config = {"frozen": True, "extra": "allow"}


class ToolDefinitionInput(BaseModel):
    """Input model for parsing tool definitions from dicts.

    Used to convert raw tool dicts from chuk_llm into ToolInfo models.
    """

    name: str
    namespace: str = "default"
    description: str | None = None
    inputSchema: dict[str, Any] = Field(default_factory=dict)
    is_async: bool = False
    tags: list[str] = Field(default_factory=list)

    model_config = {"frozen": False, "extra": "ignore"}


__all__ = [
    "ToolInputSchema",
    "ToolDefinitionInput",
]
