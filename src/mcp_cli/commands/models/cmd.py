# src/mcp_cli/commands/models/cmd.py
"""
Pydantic models for cmd action operations.

These models provide type safety for LLM messages, tool calls, and command parameters.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from .base_model import CommandBaseModel


class MessageRole(str, Enum):
    """Message role enumeration for LLM conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallFunction(CommandBaseModel):
    """Function details in a tool call."""

    name: str = Field(description="Name of the tool/function to call")
    arguments: str | dict[str, Any] = Field(
        description="Arguments as JSON string or dict"
    )


class ToolCall(CommandBaseModel):
    """Tool call made by the LLM."""

    id: str = Field(description="Unique identifier for this tool call")
    function: ToolCallFunction = Field(description="Function details")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        """Create ToolCall from dict format used by chuk-llm.

        Note: Custom implementation needed for compatibility with chuk-llm's
        nested dict format {"function": {"name": "...", "arguments": "..."}}.
        Pydantic's model_validate() expects flat structure.
        """
        return cls(
            id=data.get("id", ""),
            function=ToolCallFunction(
                name=data.get("function", {}).get("name", ""),
                arguments=data.get("function", {}).get("arguments", "{}"),
            ),
        )


class Message(CommandBaseModel):
    """A message in an LLM conversation.

    Use Pydantic native methods:
    - model_dump(exclude_none=True, mode="json") to serialize to dict
    - Message.model_validate(data) to deserialize from dict

    Note: For chuk-llm compatibility with nested tool_calls format,
    use Message.from_dict() which handles custom ToolCall conversion.
    """

    role: MessageRole = Field(description="Role of the message sender")
    content: str = Field(description="Message content")
    tool_calls: list[ToolCall] | None = Field(
        default=None, description="Tool calls made (for assistant messages)"
    )
    tool_call_id: str | None = Field(
        default=None, description="ID of the tool call (for tool messages)"
    )
    name: str | None = Field(default=None, description="Tool name (for tool messages)")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create Message from dict with custom ToolCall handling.

        Only needed when tool_calls use chuk-llm's nested format.
        For standard Pydantic dicts, use Message.model_validate(data) instead.
        """
        tool_calls = None
        if "tool_calls" in data:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]

        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content", ""),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


class LLMResponse(CommandBaseModel):
    """Response from an LLM completion."""

    response: str = Field(description="Text response from the LLM")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tool calls requested by the LLM"
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMResponse:
        """Create LLMResponse from dict format used by chuk-llm."""
        tool_calls = []
        if "tool_calls" in data and data["tool_calls"]:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]

        return cls(response=data.get("response", ""), tool_calls=tool_calls)


class CmdActionParams(CommandBaseModel):
    """Parameters for cmd action execution."""

    input_file: str | None = Field(
        default=None, description="Input file path (use '-' for stdin)"
    )
    output_file: str | None = Field(
        default=None, description="Output file path (use '-' for stdout)"
    )
    prompt: str | None = Field(default=None, description="Prompt text to use")
    tool: str | None = Field(default=None, description="Tool name to execute")
    tool_args: str | None = Field(
        default=None, description="Tool arguments as JSON string"
    )
    system_prompt: str | None = Field(default=None, description="Custom system prompt")
    raw: bool = Field(
        default=False, description="Output raw response without formatting"
    )
    single_turn: bool = Field(
        default=False, description="Disable multi-turn conversation"
    )
    max_turns: int = Field(default=30, description="Maximum conversation turns")
