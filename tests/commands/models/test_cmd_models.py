# tests/commands/models/test_cmd_models.py
"""Tests for cmd command models."""

from mcp_cli.commands.models.cmd import (
    CmdActionParams,
    LLMResponse,
    Message,
    MessageRole,
    ToolCall,
    ToolCallFunction,
)


class TestMessageRole:
    """Test MessageRole enum."""

    def test_message_role_values(self):
        """Test all MessageRole values exist."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"

    def test_message_role_is_str(self):
        """Test MessageRole is a string enum."""
        assert isinstance(MessageRole.USER, str)
        assert MessageRole.USER == "user"


class TestToolCallFunction:
    """Test ToolCallFunction model."""

    def test_creation_with_string_arguments(self):
        """Test creating with string arguments."""
        func = ToolCallFunction(name="get_weather", arguments='{"city": "NYC"}')

        assert func.name == "get_weather"
        assert func.arguments == '{"city": "NYC"}'

    def test_creation_with_dict_arguments(self):
        """Test creating with dict arguments."""
        func = ToolCallFunction(name="get_weather", arguments={"city": "NYC"})

        assert func.name == "get_weather"
        assert func.arguments == {"city": "NYC"}


class TestToolCall:
    """Test ToolCall model."""

    def test_creation(self):
        """Test creating a tool call."""
        tc = ToolCall(
            id="call_123",
            function=ToolCallFunction(name="test_tool", arguments="{}"),
        )

        assert tc.id == "call_123"
        assert tc.function.name == "test_tool"

    def test_from_dict(self):
        """Test creating from chuk-llm dict format."""
        data = {
            "id": "call_456",
            "function": {"name": "get_data", "arguments": '{"query": "test"}'},
        }

        tc = ToolCall.from_dict(data)

        assert tc.id == "call_456"
        assert tc.function.name == "get_data"
        assert tc.function.arguments == '{"query": "test"}'

    def test_from_dict_missing_fields(self):
        """Test from_dict with missing fields uses defaults."""
        data = {}

        tc = ToolCall.from_dict(data)

        assert tc.id == ""
        assert tc.function.name == ""
        assert tc.function.arguments == "{}"


class TestMessage:
    """Test Message model."""

    def test_creation_user_message(self):
        """Test creating a user message."""
        msg = Message(role=MessageRole.USER, content="Hello")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_creation_assistant_with_tool_calls(self):
        """Test creating assistant message with tool calls."""
        tc = ToolCall(
            id="call_123",
            function=ToolCallFunction(name="test", arguments="{}"),
        )
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[tc],
        )

        assert msg.role == MessageRole.ASSISTANT
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_123"

    def test_creation_tool_message(self):
        """Test creating a tool response message."""
        msg = Message(
            role=MessageRole.TOOL,
            content='{"result": "success"}',
            tool_call_id="call_123",
            name="test_tool",
        )

        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "call_123"
        assert msg.name == "test_tool"

    def test_from_dict_simple(self):
        """Test from_dict with simple message."""
        data = {"role": "user", "content": "Hello world"}

        msg = Message.from_dict(data)

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello world"

    def test_from_dict_with_tool_calls(self):
        """Test from_dict with tool calls."""
        data = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}},
                {"id": "call_2", "function": {"name": "tool2", "arguments": "{}"}},
            ],
        }

        msg = Message.from_dict(data)

        assert msg.role == MessageRole.ASSISTANT
        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0].function.name == "tool1"
        assert msg.tool_calls[1].function.name == "tool2"

    def test_from_dict_tool_message(self):
        """Test from_dict with tool response message."""
        data = {
            "role": "tool",
            "content": "result data",
            "tool_call_id": "call_123",
            "name": "my_tool",
        }

        msg = Message.from_dict(data)

        assert msg.role == MessageRole.TOOL
        assert msg.content == "result data"
        assert msg.tool_call_id == "call_123"
        assert msg.name == "my_tool"

    def test_from_dict_missing_content(self):
        """Test from_dict with missing content defaults to empty string."""
        data = {"role": "user"}

        msg = Message.from_dict(data)

        assert msg.content == ""


class TestLLMResponse:
    """Test LLMResponse model."""

    def test_creation_simple(self):
        """Test creating a simple response."""
        resp = LLMResponse(response="Hello!")

        assert resp.response == "Hello!"
        assert resp.tool_calls == []

    def test_creation_with_tool_calls(self):
        """Test creating response with tool calls."""
        tc = ToolCall(
            id="call_123",
            function=ToolCallFunction(name="test", arguments="{}"),
        )
        resp = LLMResponse(response="", tool_calls=[tc])

        assert resp.response == ""
        assert len(resp.tool_calls) == 1

    def test_from_dict_simple(self):
        """Test from_dict with simple response."""
        data = {"response": "Hello world"}

        resp = LLMResponse.from_dict(data)

        assert resp.response == "Hello world"
        assert resp.tool_calls == []

    def test_from_dict_with_tool_calls(self):
        """Test from_dict with tool calls."""
        data = {
            "response": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}},
            ],
        }

        resp = LLMResponse.from_dict(data)

        assert resp.response == ""
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].function.name == "tool1"

    def test_from_dict_empty_tool_calls(self):
        """Test from_dict with empty tool_calls list."""
        data = {"response": "test", "tool_calls": []}

        resp = LLMResponse.from_dict(data)

        assert resp.tool_calls == []

    def test_from_dict_missing_response(self):
        """Test from_dict with missing response defaults to empty string."""
        data = {}

        resp = LLMResponse.from_dict(data)

        assert resp.response == ""


class TestCmdActionParams:
    """Test CmdActionParams model."""

    def test_default_params(self):
        """Test default parameter values."""
        params = CmdActionParams()

        assert params.input_file is None
        assert params.output_file is None
        assert params.prompt is None
        assert params.tool is None
        assert params.tool_args is None
        assert params.system_prompt is None
        assert params.raw is False
        assert params.single_turn is False
        assert params.max_turns == 30

    def test_custom_params(self):
        """Test custom parameter values."""
        params = CmdActionParams(
            input_file="input.txt",
            output_file="output.txt",
            prompt="Do something",
            tool="my_tool",
            tool_args='{"arg": "value"}',
            system_prompt="You are helpful",
            raw=True,
            single_turn=True,
            max_turns=10,
        )

        assert params.input_file == "input.txt"
        assert params.output_file == "output.txt"
        assert params.prompt == "Do something"
        assert params.tool == "my_tool"
        assert params.tool_args == '{"arg": "value"}'
        assert params.system_prompt == "You are helpful"
        assert params.raw is True
        assert params.single_turn is True
        assert params.max_turns == 10
