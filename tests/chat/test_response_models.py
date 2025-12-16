# tests/chat/test_response_models.py
"""Tests for chat/response_models.py."""

from mcp_cli.chat.response_models import (
    CompletionResponse,
    FunctionCall,
    Message,
    MessageField,
    MessageRole,
    ToolCall,
    convert_messages_to_dicts,
    convert_messages_to_models,
)


class TestMessageField:
    """Tests for MessageField enum."""

    def test_values(self):
        """Test enum values."""
        assert MessageField.ROLE == "role"
        assert MessageField.CONTENT == "content"
        assert MessageField.TOOL_CALLS == "tool_calls"
        assert MessageField.TOOL_CALL_ID == "tool_call_id"
        assert MessageField.NAME == "name"


class TestCompletionResponse:
    """Tests for CompletionResponse model."""

    def test_create_simple(self):
        """Test creating a simple response."""
        resp = CompletionResponse(response="Hello!")
        assert resp.response == "Hello!"
        assert resp.tool_calls == []
        assert resp.reasoning_content is None

    def test_create_with_tool_calls(self):
        """Test creating response with tool calls."""
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 18}'),
        )
        resp = CompletionResponse(response="", tool_calls=[tool_call])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].function.name == "sqrt"

    def test_from_dict_simple(self):
        """Test from_dict with simple response."""
        data = {"response": "Hello!", "chunks_received": 5}
        resp = CompletionResponse.from_dict(data)
        assert resp.response == "Hello!"
        assert resp.chunks_received == 5

    def test_from_dict_with_tool_calls_as_dicts(self):
        """Test from_dict with tool calls as dicts."""
        data = {
            "response": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "sqrt", "arguments": '{"x": 18}'},
                }
            ],
        }
        resp = CompletionResponse.from_dict(data)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].function.name == "sqrt"

    def test_from_dict_with_tool_calls_as_models(self):
        """Test from_dict with tool calls already as ToolCall models."""
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 18}'),
        )
        data = {"response": "", "tool_calls": [tool_call]}
        resp = CompletionResponse.from_dict(data)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].function.name == "sqrt"

    def test_from_dict_empty_tool_calls(self):
        """Test from_dict with empty tool_calls list."""
        data = {"response": "Hi", "tool_calls": []}
        resp = CompletionResponse.from_dict(data)
        assert resp.tool_calls == []

    def test_from_dict_defaults(self):
        """Test from_dict with missing fields uses defaults."""
        data = {}
        resp = CompletionResponse.from_dict(data)
        assert resp.response == ""
        assert resp.tool_calls == []
        assert resp.chunks_received == 0
        assert resp.elapsed_time == 0.0
        assert resp.interrupted is False
        assert resp.streaming is False

    def test_to_dict(self):
        """Test to_dict serialization."""
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 18}'),
        )
        resp = CompletionResponse(
            response="Result",
            tool_calls=[tool_call],
            reasoning_content="Thinking...",
            chunks_received=10,
            elapsed_time=1.5,
            interrupted=False,
            streaming=True,
        )
        d = resp.to_dict()
        assert d["response"] == "Result"
        assert len(d["tool_calls"]) == 1
        assert d["reasoning_content"] == "Thinking..."
        assert d["chunks_received"] == 10
        assert d["elapsed_time"] == 1.5
        assert d["streaming"] is True

    def test_has_tool_calls_true(self):
        """Test has_tool_calls property when True."""
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments="{}"),
        )
        resp = CompletionResponse(tool_calls=[tool_call])
        assert resp.has_tool_calls is True

    def test_has_tool_calls_false(self):
        """Test has_tool_calls property when False."""
        resp = CompletionResponse(response="Hello")
        assert resp.has_tool_calls is False

    def test_has_content_true(self):
        """Test has_content property when True."""
        resp = CompletionResponse(response="Hello")
        assert resp.has_content is True

    def test_has_content_false(self):
        """Test has_content property when False."""
        resp = CompletionResponse(response="")
        assert resp.has_content is False


class TestConvertMessages:
    """Tests for message conversion functions."""

    def test_convert_messages_to_models_from_dicts(self):
        """Test converting dict messages to models."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        models = convert_messages_to_models(messages)
        assert len(models) == 2
        assert models[0].role == MessageRole.USER
        assert models[0].content == "Hello"
        assert models[1].role == MessageRole.ASSISTANT

    def test_convert_messages_to_models_passthrough(self):
        """Test that Message instances pass through unchanged."""
        msg = Message(role=MessageRole.USER, content="Hello")
        models = convert_messages_to_models([msg])
        assert models[0] is msg  # Same instance

    def test_convert_messages_to_dicts(self):
        """Test converting Message models to dicts."""
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi!"),
        ]
        dicts = convert_messages_to_dicts(messages)
        assert len(dicts) == 2
        assert dicts[0]["role"] == "user"
        assert dicts[0]["content"] == "Hello"
