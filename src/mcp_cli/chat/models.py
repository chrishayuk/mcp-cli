"""Chat-specific Pydantic models."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message roles in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class FunctionCall(BaseModel):
    """Function call within a tool call (OpenAI format)."""

    name: str = Field(description="Function/tool name")
    arguments: str = Field(description="JSON string of arguments")

    model_config = {"frozen": False}

    def get_arguments_dict(self) -> dict[str, Any]:
        """Parse arguments JSON string to dict."""
        try:
            result = json.loads(self.arguments)
            return dict(result) if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            return {}

    @classmethod
    def from_dict_args(cls, name: str, arguments: dict[str, Any]) -> "FunctionCall":
        """Create FunctionCall from dict arguments."""
        return cls(name=name, arguments=json.dumps(arguments))


class ToolCallData(BaseModel):
    """Tool call data structure (OpenAI format)."""

    id: str = Field(description="Tool call ID")
    type: str = Field(default="function", description="Type of tool call")
    function: FunctionCall = Field(description="Function call data")

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for API."""
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCallData":
        """Create from dict."""
        return cls(
            id=data["id"],
            type=data.get("type", "function"),
            function=FunctionCall(
                name=data["function"]["name"],
                arguments=data["function"]["arguments"],
            ),
        )


class Message(BaseModel):
    """A message in the conversation history."""

    role: MessageRole = Field(
        description="Message role (user, assistant, system, tool)"
    )
    content: str | None = Field(default=None, description="Message content")
    name: str | None = Field(default=None, description="Name (for tool messages)")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls (for assistant messages with tools)"
    )
    tool_call_id: str | None = Field(
        default=None, description="Tool call ID (for tool response messages)"
    )

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM API calls."""
        return self.model_dump(exclude_none=True, mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create from dict."""
        return cls.model_validate(data)

    def get_tool_calls_typed(self) -> list[ToolCallData]:
        """Get tool calls as typed ToolCallData objects."""
        if not self.tool_calls:
            return []
        return [ToolCallData.from_dict(tc) for tc in self.tool_calls]

    @classmethod
    def with_tool_calls(
        cls,
        role: MessageRole,
        tool_calls: list[ToolCallData],
        content: str | None = None,
    ) -> "Message":
        """Create a message with typed tool calls."""
        return cls(
            role=role,
            content=content,
            tool_calls=[tc.to_dict() for tc in tool_calls],
        )


class ToolExecutionRecord(BaseModel):
    """Record of a tool execution in the chat history."""

    tool_name: str = Field(description="Name of the tool that was executed")
    server: str | None = Field(
        default=None, description="Server that provided the tool"
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool"
    )
    result: Any | None = Field(default=None, description="Result from tool execution")
    error: str | None = Field(default=None, description="Error message if failed")
    execution_time: float | None = Field(
        default=None, description="Execution time in seconds"
    )
    timestamp: str | None = Field(
        default=None, description="ISO timestamp of execution"
    )

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return self.model_dump(exclude_none=True, mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolExecutionRecord:
        """Create from dict."""
        return cls.model_validate(data)


class ToolExecutionState(BaseModel):
    """State tracking for a tool currently being executed."""

    name: str = Field(description="Tool name")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )
    start_time: float = Field(description="Execution start time (timestamp)")
    result: str | None = Field(default=None, description="Tool execution result")
    success: bool = Field(default=True, description="Whether execution succeeded")
    elapsed: float | None = Field(default=None, description="Elapsed time in seconds")
    completed: bool = Field(default=False, description="Whether execution is complete")

    model_config = {"frozen": False}

    def elapsed_time(self, current_time: float) -> float:
        """Calculate elapsed time from start."""
        return current_time - self.start_time


class ChatStatus(BaseModel):
    """Status summary of the chat context."""

    provider: str = Field(description="Current LLM provider")
    model: str = Field(description="Current model")
    tool_count: int = Field(default=0, description="Number of available tools")
    internal_tool_count: int = Field(default=0, description="Number of internal tools")
    server_count: int = Field(default=0, description="Number of connected servers")
    message_count: int = Field(default=0, description="Messages in conversation")
    tool_execution_count: int = Field(
        default=0, description="Tools executed in this session"
    )

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return self.model_dump(mode="json")
