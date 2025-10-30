# mcp_cli/tools/__init__.py
"""Tools package with Pydantic models."""

from mcp_cli.tools.models import (
    # Tool and Server models
    ToolInfo,
    ServerInfo,
    ToolCallResult,
    ResourceInfo,
    # Conversation models (NEW)
    Message,
    MessageRole,
    ToolCall,
    ConversationHistory,
    TokenUsageStats,
)

__all__ = [
    # Tool and Server models
    "ToolInfo",
    "ServerInfo",
    "ToolCallResult",
    "ResourceInfo",
    # Conversation models (NEW)
    "Message",
    "MessageRole",
    "ToolCall",
    "ConversationHistory",
    "TokenUsageStats",
]
