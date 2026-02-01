"""Pydantic models for LLM content blocks - no more dict goop!"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class ContentBlockType(str, Enum):
    """Content block types - no magic strings!"""

    TEXT = "text"
    IMAGE = "image"


class TextContent(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str

    model_config = {"frozen": True}


class ImageContent(BaseModel):
    """Image content block."""

    type: Literal["image"] = "image"
    source: dict[str, str]  # Could be further typed if needed

    model_config = {"frozen": True}


# Type alias for content blocks
ContentBlock = TextContent | ImageContent | dict[str, str]  # dict for flexibility


__all__ = [
    "ContentBlockType",
    "TextContent",
    "ImageContent",
    "ContentBlock",
]
