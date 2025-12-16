# tests/chat/test_models.py
"""Tests for chat/models.py."""

import json

from mcp_cli.chat.models import (
    ChatStatus,
    FunctionCall,
    Message,
    MessageField,
    MessageRole,
    ToolCallData,
    ToolCallField,
    ToolExecutionRecord,
    ToolExecutionState,
)


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_values(self):
        """Test enum values."""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.TOOL == "tool"


class TestMessageField:
    """Tests for MessageField enum."""

    def test_values(self):
        """Test enum values."""
        assert MessageField.ROLE == "role"
        assert MessageField.CONTENT == "content"
        assert MessageField.TOOL_CALLS == "tool_calls"


class TestToolCallField:
    """Tests for ToolCallField enum."""

    def test_values(self):
        """Test enum values."""
        assert ToolCallField.ID == "id"
        assert ToolCallField.TYPE == "type"
        assert ToolCallField.FUNCTION == "function"
        assert ToolCallField.NAME == "name"
        assert ToolCallField.ARGUMENTS == "arguments"


class TestFunctionCall:
    """Tests for FunctionCall model."""

    def test_create(self):
        """Test creating a FunctionCall."""
        fc = FunctionCall(name="sqrt", arguments='{"x": 18}')
        assert fc.name == "sqrt"
        assert fc.arguments == '{"x": 18}'

    def test_get_arguments_dict(self):
        """Test parsing arguments to dict."""
        fc = FunctionCall(name="sqrt", arguments='{"x": 18}')
        args = fc.get_arguments_dict()
        assert args == {"x": 18}

    def test_get_arguments_dict_invalid_json(self):
        """Test get_arguments_dict with invalid JSON."""
        fc = FunctionCall(name="sqrt", arguments="not-json")
        args = fc.get_arguments_dict()
        assert args == {}

    def test_get_arguments_dict_non_dict(self):
        """Test get_arguments_dict with non-dict JSON."""
        fc = FunctionCall(name="sqrt", arguments="[1, 2, 3]")
        args = fc.get_arguments_dict()
        assert args == {}

    def test_from_dict_args(self):
        """Test creating from dict arguments."""
        fc = FunctionCall.from_dict_args("sqrt", {"x": 18})
        assert fc.name == "sqrt"
        assert json.loads(fc.arguments) == {"x": 18}


class TestToolCallData:
    """Tests for ToolCallData model."""

    def test_create(self):
        """Test creating ToolCallData."""
        fc = FunctionCall(name="sqrt", arguments='{"x": 18}')
        tc = ToolCallData(id="call_123", type="function", function=fc)
        assert tc.id == "call_123"
        assert tc.type == "function"
        assert tc.function.name == "sqrt"

    def test_to_dict(self):
        """Test converting to dict."""
        fc = FunctionCall(name="sqrt", arguments='{"x": 18}')
        tc = ToolCallData(id="call_123", type="function", function=fc)
        d = tc.to_dict()
        assert d["id"] == "call_123"
        assert d["type"] == "function"
        assert d["function"]["name"] == "sqrt"

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "id": "call_123",
            "type": "function",
            "index": 0,
            "function": {
                "name": "sqrt",
                "arguments": '{"x": 18}',
            },
        }
        tc = ToolCallData.from_dict(data)
        assert tc.id == "call_123"
        assert tc.function.name == "sqrt"

    def test_from_dict_defaults(self):
        """Test from_dict with missing fields."""
        data = {}
        tc = ToolCallData.from_dict(data)
        assert tc.id == ""
        assert tc.type == "function"
        assert tc.index == 0

    def test_merge_chunk_name(self):
        """Test merging chunk with name."""
        fc1 = FunctionCall(name="", arguments="")
        tc1 = ToolCallData(id="call_123", function=fc1)

        fc2 = FunctionCall(name="sqrt", arguments="")
        tc2 = ToolCallData(id="call_123", function=fc2)

        tc1.merge_chunk(tc2)
        assert tc1.function.name == "sqrt"

    def test_merge_chunk_arguments(self):
        """Test merging chunk with arguments."""
        fc1 = FunctionCall(name="sqrt", arguments='{"x":')
        tc1 = ToolCallData(id="call_123", function=fc1)

        fc2 = FunctionCall(name="", arguments=" 18}")
        tc2 = ToolCallData(id="call_123", function=fc2)

        tc1.merge_chunk(tc2)
        assert tc1.function.arguments == '{"x": 18}'


class TestMessage:
    """Tests for Message model."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_to_dict_simple(self):
        """Test to_dict with simple message."""
        msg = Message(role=MessageRole.USER, content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"

    def test_to_dict_assistant_with_tool_calls(self):
        """Test to_dict ensures content for assistant with tool_calls."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            tool_calls=[{"id": "call_1", "type": "function", "function": {}}],
        )
        d = msg.to_dict()
        # Should have content field even if None
        assert "content" in d
        assert d["content"] is None

    def test_to_dict_with_reasoning_content(self):
        """Test to_dict includes reasoning_content when set."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Answer",
            reasoning_content="I thought about this...",
        )
        d = msg.to_dict()
        assert d["reasoning_content"] == "I thought about this..."

    def test_from_dict(self):
        """Test creating from dict."""
        data = {"role": "user", "content": "Hello"}
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_get_tool_calls_typed(self):
        """Test get_tool_calls_typed."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "sqrt", "arguments": "{}"},
                }
            ],
        )
        typed = msg.get_tool_calls_typed()
        assert len(typed) == 1
        assert typed[0].id == "call_1"
        assert typed[0].function.name == "sqrt"

    def test_get_tool_calls_typed_empty(self):
        """Test get_tool_calls_typed with no tool calls."""
        msg = Message(role=MessageRole.USER, content="Hello")
        typed = msg.get_tool_calls_typed()
        assert typed == []

    def test_with_tool_calls(self):
        """Test creating message with typed tool calls."""
        fc = FunctionCall(name="sqrt", arguments='{"x": 18}')
        tc = ToolCallData(id="call_1", function=fc)

        msg = Message.with_tool_calls(
            role=MessageRole.ASSISTANT, tool_calls=[tc], content="Calling sqrt"
        )
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Calling sqrt"
        assert len(msg.tool_calls) == 1


class TestToolExecutionRecord:
    """Tests for ToolExecutionRecord model."""

    def test_create(self):
        """Test creating a record."""
        record = ToolExecutionRecord(
            tool_name="sqrt",
            arguments={"x": 18},
            result=4.2426,
        )
        assert record.tool_name == "sqrt"
        assert record.result == 4.2426

    def test_to_dict(self):
        """Test to_dict excludes None fields."""
        record = ToolExecutionRecord(tool_name="sqrt", result=4.2426)
        d = record.to_dict()
        assert "tool_name" in d
        assert "error" not in d  # None excluded

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "tool_name": "sqrt",
            "arguments": {"x": 18},
            "result": 4.2426,
        }
        record = ToolExecutionRecord.from_dict(data)
        assert record.tool_name == "sqrt"
        assert record.result == 4.2426


class TestToolExecutionState:
    """Tests for ToolExecutionState model."""

    def test_create(self):
        """Test creating state."""
        state = ToolExecutionState(name="sqrt", arguments={"x": 18}, start_time=1000.0)
        assert state.name == "sqrt"
        assert state.start_time == 1000.0
        assert state.completed is False

    def test_elapsed_time(self):
        """Test elapsed_time calculation."""
        state = ToolExecutionState(name="sqrt", arguments={}, start_time=1000.0)
        elapsed = state.elapsed_time(1005.0)
        assert elapsed == 5.0


class TestChatStatus:
    """Tests for ChatStatus model."""

    def test_create(self):
        """Test creating status."""
        status = ChatStatus(
            provider="openai",
            model="gpt-4",
            tool_count=10,
            server_count=2,
        )
        assert status.provider == "openai"
        assert status.model == "gpt-4"
        assert status.tool_count == 10

    def test_to_dict(self):
        """Test to_dict."""
        status = ChatStatus(provider="openai", model="gpt-4")
        d = status.to_dict()
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4"
