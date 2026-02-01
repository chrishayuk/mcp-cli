# tests/mcp_cli/chat/test_tool_processor.py
import json
import pytest
from datetime import datetime, UTC

from chuk_tool_processor import ToolResult as CTPToolResult
import chuk_ai_session_manager.guards.manager as _guard_mgr
from chuk_ai_session_manager.guards import (
    get_tool_state,
    reset_tool_state,
    RuntimeLimits,
    ToolStateManager,
)

from mcp_cli.chat.tool_processor import ToolProcessor
from mcp_cli.chat.response_models import ToolCall, FunctionCall
from mcp_cli.tools.models import ToolCallResult


@pytest.fixture(autouse=True)
def _fresh_tool_state():
    """Reset the global tool state singleton before each test with permissive limits."""
    reset_tool_state()
    _guard_mgr._tool_state = ToolStateManager(
        limits=RuntimeLimits(
            per_tool_cap=100,
            tool_budget_total=100,
            discovery_budget=50,
            execution_budget=50,
        )
    )
    yield
    reset_tool_state()


# ---------------------------
# Dummy classes for testing
# ---------------------------


class DummyUIManager:
    def __init__(self):
        self.printed_calls = []  # To record calls to print_tool_call
        self.is_streaming_response = False  # Add missing attribute

    def print_tool_call(self, tool_name, raw_arguments):
        self.printed_calls.append((tool_name, raw_arguments))

    async def finish_tool_execution(self, result=None, success=True):
        # Add async method that tool processor expects
        pass

    def do_confirm_tool_execution(self, tool_name, arguments):
        # Mock confirmation - always return True for tests
        return True

    async def start_tool_execution(self, tool_name, arguments):
        # Mock start tool execution - no-op for tests
        pass


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
    """Mock tool manager with execute_tool and stream_execute_tools methods."""

    def __init__(self, return_result=None, raise_exception=False):
        self.return_result = return_result or {
            "isError": False,
            "content": "Tool executed successfully",
        }
        self.raise_exception = raise_exception
        self.executed_tool = None
        self.executed_args = None

    async def execute_tool(self, tool_name, arguments, namespace=None, timeout=None):
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

    async def stream_execute_tools(
        self, calls, timeout=None, on_tool_start=None, max_concurrency=4
    ):
        """Yield CTPToolResult for each call."""
        import platform
        import os

        for call in calls:
            self.executed_tool = call.tool
            self.executed_args = call.arguments

            # Invoke start callback if provided
            if on_tool_start:
                await on_tool_start(call)

            if self.raise_exception:
                now = datetime.now(UTC)
                yield CTPToolResult(
                    id=call.id,
                    tool=call.tool,
                    result=None,
                    error="Simulated execute_tool exception",
                    start_time=now,
                    end_time=now,
                    machine=platform.node(),
                    pid=os.getpid(),
                )
            elif self.return_result.get("isError"):
                now = datetime.now(UTC)
                yield CTPToolResult(
                    id=call.id,
                    tool=call.tool,
                    result=None,
                    error=self.return_result.get("error", "Simulated error"),
                    start_time=now,
                    end_time=now,
                    machine=platform.node(),
                    pid=os.getpid(),
                )
            else:
                now = datetime.now(UTC)
                yield CTPToolResult(
                    id=call.id,
                    tool=call.tool,
                    result=self.return_result.get("content"),
                    error=None,
                    start_time=now,
                    end_time=now,
                    machine=platform.node(),
                    pid=os.getpid(),
                )


class DummyContext:
    """A dummy context object with conversation_history and managers."""

    def __init__(self, stream_manager=None, tool_manager=None):
        self.conversation_history = []
        self.stream_manager = stream_manager
        self.tool_manager = tool_manager

    def inject_tool_message(self, message):
        """Add a message to conversation history (matches ChatContext API)."""
        self.conversation_history.append(message)


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

    # Create a proper ToolCall Pydantic model instead of a dict
    tool_call = ToolCall(
        id="call_echo",
        type="function",
        function=FunctionCall(name="echo", arguments='{"msg": "Hello"}'),
    )
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
    assert call_record.tool_calls is not None
    # tool_calls is now a list of ToolCall Pydantic models, not dicts
    assert any(item.id == "call_echo" for item in call_record.tool_calls)

    # Verify the response record.
    assert response_record.role.value == "tool"
    # Content now includes value binding info ($vN = value)
    assert "Tool executed successfully" in response_record.content


@pytest.mark.asyncio
async def test_process_tool_calls_with_argument_parsing():
    # Test that raw arguments given as a JSON string are parsed into a dict.
    result_dict = {"isError": False, "content": {"parsed": True}}
    stream_manager = DummyStreamManager(return_result=result_dict)
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(stream_manager=stream_manager, tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    # Register 123 as a user-provided literal so it passes ungrounded check
    tool_state = get_tool_state()
    tool_state.register_user_literals("Test with value 123")

    tool_call = {
        "function": {"name": "parse_tool", "arguments": '{"num": 123}'},
        "id": "call_parse",
    }
    await processor.process_tool_calls([tool_call])

    # Check that execute_tool was given parsed arguments (a dict).
    assert isinstance(tool_manager.executed_args, dict)
    assert tool_manager.executed_args.get("num") == 123

    # Check that the response record content contains the formatted result.
    # Note: Content now includes value binding info ($vN = value) appended
    response_record = context.conversation_history[1]
    expected_formatted = json.dumps(result_dict["content"], indent=2)
    assert expected_formatted in response_record.content


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
    assert response_record.role.value == "tool"
    assert "Error: Simulated error" in response_record.content


@pytest.mark.asyncio
async def test_process_tool_calls_no_tool_manager():
    # Test when no tool manager is available.
    context = DummyContext(stream_manager=None, tool_manager=None)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    # Supply a dummy tool call - pass as individual dict, not wrapped in list
    tool_call = {
        "function": {"name": "dummy_tool", "arguments": '{"key": "value"}'},
        "id": "test1",
    }

    # Pass as a list to process_tool_calls - should raise RuntimeError
    with pytest.raises(RuntimeError, match="No tool manager available"):
        await processor.process_tool_calls([tool_call])


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
        if entry.role.value == "tool" and entry.content and "Error:" in entry.content
    ]
    assert len(error_entries) >= 1
    # The error should contain the exception message
    assert any("Simulated execute_tool exception" in e.content for e in error_entries)
