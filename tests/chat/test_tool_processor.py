# tests/mcp_cli/chat/test_tool_processor.py
import json
import pytest

from mcp_cli.chat.tool_processor import ToolProcessor
from mcp_cli.tools.models import ToolCallResult

# ---------------------------
# Dummy classes for testing
# ---------------------------


class DummyUIManager:
    def __init__(self):
        self.printed_calls = []  # To record calls to print_tool_call
        self.is_streaming_response = False  # Add missing attribute

    def print_tool_call(self, tool_name, raw_arguments):
        self.printed_calls.append((tool_name, raw_arguments))

    def finish_tool_execution(self, result=None, success=True):
        # Add method that tool processor expects
        pass

    def do_confirm_tool_execution(self, tool_name, arguments):
        # Mock confirmation - always return True for tests
        return True


class DummyStreamManager:
    def __init__(self, return_result=None, raise_exception=False):
        # return_result: dictionary to return when call_tool is invoked.
        self.return_result = return_result or {
            "isError": False,
            "content": "Successful call",
        }
        self.raise_exception = raise_exception
        self.called_tool = None
        self.called_args = None

    async def call_tool(self, tool_name, arguments):
        self.called_tool = tool_name
        self.called_args = arguments
        if self.raise_exception:
            raise Exception("Simulated call_tool exception")
        return self.return_result


class DummyToolManager:
    """Mock tool manager with execute_tool method that returns ToolCallResult."""

    def __init__(self, return_result=None, raise_exception=False):
        self.return_result = return_result or {
            "isError": False,
            "content": "Tool executed successfully",
        }
        self.raise_exception = raise_exception
        self.executed_tool = None
        self.executed_args = None

    async def execute_tool(self, tool_name, arguments):
        self.executed_tool = tool_name
        self.executed_args = arguments
        if self.raise_exception:
            raise Exception("Simulated execute_tool exception")

        # Return a ToolCallResult object, not a dict
        if self.return_result.get("isError"):
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=self.return_result.get("error", "Simulated error"),
            )
        else:
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=self.return_result.get("content"),
                error=None,
            )


class DummyContext:
    """A dummy context object with conversation_history and managers."""

    def __init__(self, stream_manager=None, tool_manager=None):
        self.conversation_history = []
        self.stream_manager = stream_manager
        self.tool_manager = tool_manager


# ---------------------------
# Tests for ToolProcessor
# ---------------------------


@pytest.mark.asyncio
async def test_process_tool_calls_empty_list(capfd):
    # Test that an empty list of tool_calls prints a warning and does nothing.
    tool_manager = DummyToolManager()
    context = DummyContext(
        stream_manager=DummyStreamManager(), tool_manager=tool_manager
    )
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    await processor.process_tool_calls([])
    # No tool calls processed; conversation history remains unchanged.
    assert context.conversation_history == []

    # Optionally, also check that a warning was printed.
    captured = capfd.readouterr().out
    assert "Empty tool_calls list received." in captured


@pytest.mark.asyncio
async def test_process_tool_calls_successful_tool():
    # Test a successful tool call.
    result_dict = {"isError": False, "content": "Tool executed successfully"}
    stream_manager = DummyStreamManager(return_result=result_dict)
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(stream_manager=stream_manager, tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_call = {
        "function": {"name": "echo", "arguments": '{"msg": "Hello"}'},
        "id": "call_echo",
    }
    await processor.process_tool_calls([tool_call])

    # Verify that the UI manager printed the tool call.
    assert ("echo", '{"msg": "Hello"}') in ui_manager.printed_calls

    # Expect two conversation history records:
    #  - First: an assistant record containing the tool call details.
    #  - Second: a tool record containing the result.
    assert len(context.conversation_history) == 2

    call_record = context.conversation_history[0]
    response_record = context.conversation_history[1]

    # Verify the tool call record contains the correct id.
    assert "tool_calls" in call_record
    assert any(item.get("id") == "call_echo" for item in call_record["tool_calls"])

    # Verify the response record.
    assert response_record["role"] == "tool"
    assert response_record["content"] == "Tool executed successfully"


@pytest.mark.asyncio
async def test_process_tool_calls_with_argument_parsing():
    # Test that raw arguments given as a JSON string are parsed into a dict.
    result_dict = {"isError": False, "content": {"parsed": True}}
    stream_manager = DummyStreamManager(return_result=result_dict)
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(stream_manager=stream_manager, tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_call = {
        "function": {"name": "parse_tool", "arguments": '{"num": 123}'},
        "id": "call_parse",
    }
    await processor.process_tool_calls([tool_call])

    # Check that execute_tool was given parsed arguments (a dict).
    assert isinstance(tool_manager.executed_args, dict)
    assert tool_manager.executed_args.get("num") == 123

    # Check that the response record content is formatted as a JSON string.
    response_record = context.conversation_history[1]
    expected_formatted = json.dumps(result_dict["content"], indent=2)
    assert response_record["content"] == expected_formatted


@pytest.mark.asyncio
async def test_process_tool_calls_tool_call_error():
    # Test a tool call that returns an error result.
    error_result = {
        "isError": True,
        "error": "Simulated error",
        "content": "Error: Simulated error",
    }
    stream_manager = DummyStreamManager(return_result=error_result)
    tool_manager = DummyToolManager(return_result=error_result)
    context = DummyContext(stream_manager=stream_manager, tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_call = {
        "function": {"name": "fail_tool", "arguments": '{"dummy": "data"}'},
        "id": "fail_call",
    }
    await processor.process_tool_calls([tool_call])

    # Conversation history should include a tool record with an error message.
    response_record = context.conversation_history[1]
    assert response_record["role"] == "tool"
    assert "Error: Simulated error" in response_record["content"]


@pytest.mark.asyncio
async def test_process_tool_calls_no_stream_manager(capfd):
    # Test when no stream manager is available.
    context = DummyContext(stream_manager=None, tool_manager=None)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    # Supply a dummy tool call - pass as individual dict, not wrapped in list
    tool_call = {
        "function": {"name": "dummy_tool", "arguments": '{"key": "value"}'},
        "id": "test1",
    }

    # Pass as a list to process_tool_calls
    await processor.process_tool_calls([tool_call])

    # The actual error message is "No tool manager available for tool execution"
    error_msgs = [
        entry.get("content", "")
        for entry in context.conversation_history
        if entry.get("content") is not None
    ]
    assert any("No tool manager available" in msg for msg in error_msgs)


@pytest.mark.asyncio
async def test_process_tool_calls_exception_in_call():
    # Test that an exception raised during execute_tool is caught and an error is recorded.
    stream_manager = DummyStreamManager()
    tool_manager = DummyToolManager(raise_exception=True)
    context = DummyContext(stream_manager=stream_manager, tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_call = {
        "function": {"name": "exception_tool", "arguments": '{"dummy": "data"}'},
        "id": "exc_call",
    }
    await processor.process_tool_calls([tool_call])

    # Look for an error entry
    error_entries = [
        entry
        for entry in context.conversation_history
        if entry.get("role") == "tool" and "Error:" in entry.get("content", "")
    ]
    assert len(error_entries) >= 1
    # The error should contain the exception message
    assert any(
        "Simulated execute_tool exception" in e["content"] for e in error_entries
    )
