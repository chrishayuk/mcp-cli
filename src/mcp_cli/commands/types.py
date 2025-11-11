# src/mcp_cli/commands/types.py
"""Type aliases for common command types."""

from __future__ import annotations


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
ServerList = list["ServerInfoResponse"]
ResourceList = list["ResourceInfoResponse"]
PromptList = list["PromptInfoResponse"]
ToolList = list["ToolInfoResponse"]

__all__ = [
    "ServerList",
    "ResourceList",
    "PromptList",
    "ToolList",
]
