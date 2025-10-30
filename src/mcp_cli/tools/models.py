# mcp_cli/tools/models.py
"""Data models used throughout MCP-CLI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Tool-related models (converted to Pydantic)
# ──────────────────────────────────────────────────────────────────────────────
class ToolInfo(BaseModel):
    """Information about a tool."""

    name: str
    namespace: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    is_async: bool = False
    tags: List[str] = Field(default_factory=list)
    supports_streaming: bool = False

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @property
    def fully_qualified_name(self) -> str:
        """Get the fully qualified tool name (namespace.name)."""
        return f"{self.namespace}.{self.name}" if self.namespace else self.name

    @property
    def display_name(self) -> str:
        """Get a user-friendly display name."""
        return self.name

    @property
    def has_parameters(self) -> bool:
        """Check if the tool has parameters defined."""
        return bool(self.parameters and self.parameters.get("properties"))

    @property
    def required_parameters(self) -> List[str]:
        """Get list of required parameter names."""
        if not self.parameters:
            return []
        required = self.parameters.get("required", [])
        return required if isinstance(required, list) else []

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "No description provided",
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


class ServerInfo(BaseModel):
    """Information about a connected server instance."""

    id: int
    name: str
    status: str
    tool_count: int
    namespace: str
    enabled: bool = True
    connected: bool = False
    transport: str = "stdio"  # "stdio", "http", "sse"
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None  # From server metadata
    version: Optional[str] = None  # Server version
    command: Optional[str] = None  # Server command if known
    args: List[str] = Field(default_factory=list)  # Command arguments
    env: Dict[str, str] = Field(default_factory=dict)  # Environment variables

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @property
    def is_healthy(self) -> bool:
        """Check if server is healthy and ready."""
        return self.status == "healthy" and self.connected

    @property
    def display_status(self) -> str:
        """Get a user-friendly status string."""
        if not self.enabled:
            return "disabled"
        elif not self.connected:
            return "disconnected"
        else:
            return self.status

    @property
    def display_description(self) -> str:
        """Get description or a default based on name."""
        # Use server-provided description if available
        if self.description:
            return self.description
        # Otherwise just return a generic description
        return f"{self.name} MCP server"

    @property
    def has_tools(self) -> bool:
        """Check if server has any tools."""
        return self.tool_count > 0


class ToolCallResult(BaseModel):
    """Outcome of a tool execution."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

    model_config = {"frozen": False, "arbitrary_types_allowed": True, "extra": "allow"}

    @property
    def display_result(self) -> str:
        """Get a display-friendly result string."""
        if not self.success:
            return f"Error: {self.error or 'Unknown error'}"
        elif isinstance(self.result, (dict, list)):
            import json

            return json.dumps(self.result, indent=2)
        else:
            return str(self.result)

    @property
    def has_error(self) -> bool:
        """Check if the result contains an error."""
        return not self.success or self.error is not None

    def to_conversation_history(self) -> str:
        """Format for inclusion in conversation history."""
        if self.success:
            return self.display_result
        else:
            return f"Tool execution failed: {self.error}"


# ──────────────────────────────────────────────────────────────────────────────
# Chat conversation models (Pydantic)
# ──────────────────────────────────────────────────────────────────────────────
from enum import Enum
from datetime import datetime


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """Represents a tool call in a conversation."""
    id: str
    type: str = "function"
    function: Dict[str, Any]  # Contains "name" and "arguments"

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @property
    def name(self) -> str:
        """Get the tool/function name."""
        return self.function.get("name", "")

    @property
    def arguments(self) -> Dict[str, Any]:
        """Get the tool arguments as a dict."""
        import json
        args = self.function.get("arguments", "{}")
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                return {}
        return args


class Message(BaseModel):
    """Represents a single message in a conversation."""
    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None  # For tool messages
    tool_calls: Optional[List[ToolCall]] = None  # For assistant messages with tool calls
    tool_call_id: Optional[str] = None  # For tool response messages
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for LLM APIs."""
        result: Dict[str, Any] = {"role": self.role.value}

        if self.content is not None:
            result["content"] = self.content

        if self.name is not None:
            result["name"] = self.name

        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function
                }
                for tc in self.tool_calls
            ]

        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        # Handle role conversion
        role_str = data.get("role", "user")
        role = MessageRole(role_str) if isinstance(role_str, str) else role_str

        # Handle tool_calls conversion
        tool_calls = None
        if "tool_calls" in data and data["tool_calls"]:
            tool_calls = [
                ToolCall(**tc) if isinstance(tc, dict) else tc
                for tc in data["tool_calls"]
            ]

        return cls(
            role=role,
            content=data.get("content"),
            name=data.get("name"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            timestamp=data.get("timestamp", datetime.now())
        )


class ConversationHistory(BaseModel):
    """Manages the conversation history with Pydantic validation."""
    messages: List[Message] = Field(default_factory=list)
    system_prompt: Optional[str] = None

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    def add_message(self, message: Message) -> None:
        """Add a message to the history."""
        self.messages.append(message)

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.messages.append(Message(role=MessageRole.USER, content=content))

    def add_assistant_message(self, content: str, tool_calls: Optional[List[ToolCall]] = None) -> None:
        """Add an assistant message."""
        self.messages.append(
            Message(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)
        )

    def add_tool_response(self, tool_call_id: str, content: str, name: str) -> None:
        """Add a tool response message."""
        self.messages.append(
            Message(
                role=MessageRole.TOOL,
                content=content,
                tool_call_id=tool_call_id,
                name=name
            )
        )

    def get_messages_for_llm(self) -> List[Dict[str, Any]]:
        """Get messages in the format expected by LLM APIs."""
        return [msg.to_dict() for msg in self.messages]

    def clear(self, keep_system: bool = True) -> None:
        """Clear the conversation history."""
        if keep_system and self.messages and self.messages[0].role == MessageRole.SYSTEM:
            system_msg = self.messages[0]
            self.messages = [system_msg]
        else:
            self.messages = []

    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.messages)

    @property
    def length(self) -> int:
        """Get the number of messages (excluding system prompt)."""
        count = len(self.messages)
        if count > 0 and self.messages[0].role == MessageRole.SYSTEM:
            return count - 1
        return count


class TokenUsageStats(BaseModel):
    """Token usage statistics for a conversation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    segments: int = 1

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    def update(self, prompt: int, completion: int, cost: float = 0.0) -> None:
        """Update token counts."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += (prompt + completion)
        self.estimated_cost += cost

    def approaching_limit(self, threshold: int) -> bool:
        """Check if we're approaching the token threshold."""
        return self.total_tokens >= (threshold * 0.8)  # 80% of threshold

    def exceeded_limit(self, threshold: int) -> bool:
        """Check if we've exceeded the token threshold."""
        return self.total_tokens >= threshold


# ──────────────────────────────────────────────────────────────────────────────
# NEW - resource-related models (converted to Pydantic)
# ──────────────────────────────────────────────────────────────────────────────
class ResourceInfo(BaseModel):
    """
    Canonical representation of *one* resource entry as returned by
    ``resources.list``.

    The MCP spec does not prescribe a single shape, so we normalise the common
    fields we use in the UI.  **All additional keys** are preserved inside
    ``extra``.
    """

    # Common attributes we frequently need in the UI
    id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None

    # Anything else goes here …
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    # ------------------------------------------------------------------ #
    # Factory helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_raw(cls, raw: Any) -> "ResourceInfo":
        """
        Convert a raw list item (dict | str | int | …) into a ResourceInfo.

        If *raw* is not a mapping we treat it as an opaque scalar and store it
        in ``extra["value"]`` so it is never lost.
        """
        if isinstance(raw, dict):
            known = {k: raw.get(k) for k in ("id", "name", "type")}
            extra = {k: v for k, v in raw.items() if k not in known}
            return cls(**known, extra=extra)
        # primitive - wrap it
        return cls(extra={"value": raw})
