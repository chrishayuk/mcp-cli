# src/mcp_cli/commands/models/conversation.py
"""Conversation command models."""

from __future__ import annotations

from pydantic import Field

from mcp_cli.commands.models.base_model import CommandBaseModel


class ConversationActionParams(CommandBaseModel):
    """Parameters for conversation actions."""

    action: str = Field(description="Action to perform (list/save/load/delete)")
    name: str | None = Field(default=None, description="Conversation name")
    format: str = Field(default="table", description="Output format")


class ConversationInfo(CommandBaseModel):
    """Information about a saved conversation."""

    name: str = Field(description="Conversation name")
    timestamp: str = Field(description="Last modified timestamp")
    message_count: int = Field(default=0, description="Number of messages")
    size: int = Field(default=0, description="Size in bytes")
