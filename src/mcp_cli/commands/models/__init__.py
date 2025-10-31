# src/mcp_cli/commands/models/__init__.py
"""
Pydantic models for command actions and their parameters/results.

These models provide type safety, validation, and documentation for command actions.
"""

from __future__ import annotations

# Server models
from .server import ServerActionParams, ServerStatusInfo, ServerPerformanceInfo

# Model models
from .model import ModelActionParams, ModelInfo

# Provider models
from .provider import ProviderActionParams, ProviderInfo

# Token models
from .token import (
    TokenListParams,
    TokenSetParams,
    TokenDeleteParams,
    TokenClearParams,
    TokenProviderParams,
)

# Tool models
from .tool import ToolActionParams, ToolCallParams

# Resource models
from .resource import ResourceActionParams

# Prompt models
from .prompt import PromptActionParams

# Theme models
from .theme import ThemeActionParams, ThemeInfo

# Conversation models
from .conversation import ConversationActionParams, ConversationInfo

# Response models
from .responses import (
    ServerInfoResponse,
    ResourceInfoResponse,
    PromptInfoResponse,
    ToolInfoResponse,
)

__all__ = [
    # Server models
    "ServerActionParams",
    "ServerStatusInfo",
    "ServerPerformanceInfo",
    # Model models
    "ModelActionParams",
    "ModelInfo",
    # Provider models
    "ProviderActionParams",
    "ProviderInfo",
    # Token models
    "TokenListParams",
    "TokenSetParams",
    "TokenDeleteParams",
    "TokenClearParams",
    "TokenProviderParams",
    # Tool models
    "ToolActionParams",
    "ToolCallParams",
    # Resource models
    "ResourceActionParams",
    # Prompt models
    "PromptActionParams",
    # Theme models
    "ThemeActionParams",
    "ThemeInfo",
    # Conversation models
    "ConversationActionParams",
    "ConversationInfo",
    # Response models
    "ServerInfoResponse",
    "ResourceInfoResponse",
    "PromptInfoResponse",
    "ToolInfoResponse",
]
