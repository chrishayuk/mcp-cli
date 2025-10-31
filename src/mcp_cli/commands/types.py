# src/mcp_cli/commands/types.py
"""Type aliases for common command types."""

from __future__ import annotations

from typing import List

# Import response models
# These will be imported once the models are updated
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_cli.commands.models import (
        ServerInfoResponse,
        ResourceInfoResponse,
        PromptInfoResponse,
        ToolInfoResponse,
    )

# Response list type aliases
ServerList = List["ServerInfoResponse"]
ResourceList = List["ResourceInfoResponse"]
PromptList = List["PromptInfoResponse"]
ToolList = List["ToolInfoResponse"]

__all__ = [
    "ServerList",
    "ResourceList",
    "PromptList",
    "ToolList",
]
