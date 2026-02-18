"""Clean Pydantic models for LLM responses - no dictionary goop!

All LLM response handling should use these models instead of raw dicts.

IMPORTANT: We import Message, ToolCall, FunctionCall from chuk_llm.core.models
to avoid duplicating model definitions. These are the canonical types used
across chuk-ai ecosystem.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# chuk_llm's canonical models (frozen/immutable, for finalized LLM responses).
# For SessionManager history messages, use mcp_cli.chat.models.HistoryMessage.
# For mutable streaming function calls, use mcp_cli.chat.models.FunctionCallData.
from chuk_llm.core.models import (
    FunctionCall,
    Message,
    MessageRole,
    ToolCall,
)

# Canonical MessageField enum lives in chat/models.py (has REASONING_CONTENT).
from mcp_cli.chat.models import MessageField

# Re-export for backwards compatibility
__all__ = [
    "FunctionCall",
    "Message",
    "MessageRole",
    "MessageField",
    "ToolCall",
    "CompletionResponse",
    "convert_messages_to_models",
    "convert_messages_to_dicts",
]


# ================================================================
# Completion Response Models
# ================================================================


class CompletionResponse(BaseModel):
    """LLM completion response with streaming metadata - type-safe!

    This extends the basic chuk_llm CompletionResponse with MCP-CLI-specific
    streaming metadata like chunks_received, elapsed_time, etc.

    Use this instead of raw dicts for all completion handling.
    No more completion.get('response', 'No response')!
    """

    response: str = Field(default="", description="Text response from LLM")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tool calls requested by LLM"
    )
    reasoning_content: str | None = Field(
        default=None, description="Reasoning content (if available)"
    )
    # MCP-CLI-specific streaming metadata
    chunks_received: int = Field(default=0, description="Number of chunks received")
    elapsed_time: float = Field(default=0.0, description="Response time in seconds")
    interrupted: bool = Field(default=False, description="Was streaming interrupted")
    streaming: bool = Field(default=False, description="Was this a streaming response")
    usage: dict[str, int] | None = Field(
        default=None, description="Token usage from provider"
    )

    model_config = {"frozen": True}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionResponse:
        """Create from dictionary (for legacy compatibility).

        Args:
            data: Raw completion dictionary

        Returns:
            CompletionResponse instance
        """
        # Parse tool calls if present - handle both dict and ToolCall instances
        tool_calls = []
        if "tool_calls" in data and data["tool_calls"]:
            for tc in data["tool_calls"]:
                if isinstance(tc, ToolCall):
                    tool_calls.append(tc)
                elif isinstance(tc, dict):
                    # Create ToolCall from dict using chuk_llm's model_validate
                    tool_calls.append(ToolCall.model_validate(tc))

        return cls(
            response=data.get("response", ""),
            tool_calls=tool_calls,
            reasoning_content=data.get("reasoning_content"),
            chunks_received=data.get("chunks_received", 0),
            elapsed_time=data.get("elapsed_time", 0.0),
            interrupted=data.get("interrupted", False),
            streaming=data.get("streaming", False),
            usage=data.get("usage"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (for backwards compatibility)."""
        return {
            "response": self.response,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "reasoning_content": self.reasoning_content,
            "chunks_received": self.chunks_received,
            "elapsed_time": self.elapsed_time,
            "interrupted": self.interrupted,
            "streaming": self.streaming,
            "usage": self.usage,
        }

    @property
    def has_tool_calls(self) -> bool:
        """Check if response includes tool calls."""
        return len(self.tool_calls) > 0

    @property
    def has_content(self) -> bool:
        """Check if response has text content."""
        return bool(self.response)


# ================================================================
# Helper Functions
# ================================================================


def convert_messages_to_models(messages: list[dict[str, Any]]) -> list[Message]:
    """Convert list of message dicts to Message models.

    Uses chuk_llm's Message.model_validate for proper Pydantic parsing.

    Args:
        messages: List of raw message dictionaries

    Returns:
        List of Message instances
    """
    return [
        Message.model_validate(msg) if isinstance(msg, dict) else msg
        for msg in messages
    ]


def convert_messages_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert list of Message models to dicts.

    Uses chuk_llm's Message.to_dict() for proper serialization.

    Args:
        messages: List of Message instances

    Returns:
        List of message dictionaries
    """
    return [msg.to_dict() for msg in messages]
