# tests/mcp_cli/chat/test_tool_processor.py
import json
import logging
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
async def test_process_tool_calls_empty_list(caplog):
    # Test that an empty list of tool_calls logs a warning and does nothing.
    tool_manager = DummyToolManager()
    context = DummyContext(
        stream_manager=DummyStreamManager(), tool_manager=tool_manager
    )
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    with caplog.at_level(logging.WARNING, logger="mcp_cli.chat.tool_processor"):
        await processor.process_tool_calls([])
    # No tool calls processed; conversation history remains unchanged.
    assert context.conversation_history == []

    # Check that a warning was logged.
    assert "Empty tool_calls list received." in caplog.text


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
    # The finally block still runs, so missing results are filled in.
    with pytest.raises(RuntimeError, match="No tool manager available"):
        await processor.process_tool_calls([tool_call])


@pytest.mark.asyncio
async def test_ensure_all_tool_results_fills_missing():
    """Test that _ensure_all_tool_results adds placeholder results for missing tool_call_ids."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    # Simulate: assistant message was added but no tool result was added
    processor._result_ids_added = set()
    tool_calls = [
        ToolCall(
            id="call_orphan_1",
            type="function",
            function=FunctionCall(name="tool_a", arguments='{"x": 1}'),
        ),
        ToolCall(
            id="call_orphan_2",
            type="function",
            function=FunctionCall(name="tool_b", arguments='{"y": 2}'),
        ),
    ]

    # Only tool_a got a result
    processor._result_ids_added.add("call_orphan_1")

    processor._ensure_all_tool_results(tool_calls)

    # tool_b should now have a placeholder result in the conversation history
    tool_results = [
        msg for msg in context.conversation_history if msg.role.value == "tool"
    ]
    assert len(tool_results) == 1
    assert tool_results[0].tool_call_id == "call_orphan_2"
    assert "interrupted or failed" in tool_results[0].content


@pytest.mark.asyncio
async def test_ensure_all_tool_results_noop_when_all_present():
    """Test that _ensure_all_tool_results does nothing when all results are present."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    processor._result_ids_added = {"call_1", "call_2"}
    tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="tool_a", arguments="{}"),
        ),
        ToolCall(
            id="call_2",
            type="function",
            function=FunctionCall(name="tool_b", arguments="{}"),
        ),
    ]

    processor._ensure_all_tool_results(tool_calls)

    # No new messages should have been added
    assert len(context.conversation_history) == 0


@pytest.mark.asyncio
async def test_process_tool_calls_finally_adds_missing_results():
    """Test that the finally block adds results for tools that never executed.

    When a tool_manager raises RuntimeError, the finally block should still
    ensure all tool_call_ids have results, preventing OpenAI 400 errors.
    """
    context = DummyContext(stream_manager=None, tool_manager=None)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_call = ToolCall(
        id="call_no_manager",
        type="function",
        function=FunctionCall(name="some_tool", arguments='{"a": "b"}'),
    )

    with pytest.raises(RuntimeError):
        await processor.process_tool_calls([tool_call])

    # The finally block should have added a placeholder result
    tool_results = [
        msg for msg in context.conversation_history if msg.role.value == "tool"
    ]
    assert len(tool_results) >= 1
    assert any(msg.tool_call_id == "call_no_manager" for msg in tool_results)


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


# ---------------------------
# Tests for orphaned tool_call_id safety net
# ---------------------------


class DenyConfirmUIManager(DummyUIManager):
    """UI manager that denies tool confirmation."""

    def do_confirm_tool_execution(self, tool_name, arguments):
        return False


class FailingInjectContext(DummyContext):
    """Context whose inject_tool_message raises after the first N calls."""

    def __init__(self, tool_manager=None, fail_after=1):
        super().__init__(tool_manager=tool_manager)
        self._inject_count = 0
        self._fail_after = fail_after

    def inject_tool_message(self, message):
        self._inject_count += 1
        if self._inject_count > self._fail_after:
            raise RuntimeError("Simulated inject failure")
        super().inject_tool_message(message)


@pytest.mark.asyncio
async def test_cancelled_tool_still_gets_result_for_remaining():
    """When user cancels confirmation on the 2nd tool, the 2nd tool still gets a result."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)

    call_count = [0]

    class SelectiveDenyUI(DummyUIManager):
        """Denies the second tool call."""

        def do_confirm_tool_execution(self, tool_name, arguments):
            call_count[0] += 1
            return call_count[0] <= 1  # Allow first, deny second

    ui_manager = SelectiveDenyUI()
    processor = ToolProcessor(context, ui_manager)

    tool_calls = [
        ToolCall(
            id="call_ok",
            type="function",
            function=FunctionCall(name="tool_a", arguments="{}"),
        ),
        ToolCall(
            id="call_denied",
            type="function",
            function=FunctionCall(name="tool_b", arguments="{}"),
        ),
    ]

    await processor.process_tool_calls(tool_calls)

    # Both tool_call_ids must have results (one executed, one cancelled/placeholder)
    tool_results = [
        msg for msg in context.conversation_history if msg.role.value == "tool"
    ]
    result_ids = {msg.tool_call_id for msg in tool_results}
    assert "call_ok" in result_ids or "call_denied" in result_ids
    # The key assertion: no orphaned tool_call_ids
    assistant_msgs = [
        msg
        for msg in context.conversation_history
        if msg.role.value == "assistant" and msg.tool_calls
    ]
    for amsg in assistant_msgs:
        for tc in amsg.tool_calls:
            tc_id = tc.id if hasattr(tc, "id") else tc.get("id")
            assert tc_id in result_ids, f"Orphaned tool_call_id: {tc_id}"


@pytest.mark.asyncio
async def test_inject_failure_is_caught_by_finally():
    """When inject_tool_message fails, the finally block fills missing results."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    # fail_after=1 means the assistant message succeeds, but the tool result inject fails
    context = FailingInjectContext(tool_manager=tool_manager, fail_after=1)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_call = ToolCall(
        id="call_inject_fail",
        type="function",
        function=FunctionCall(name="some_tool", arguments="{}"),
    )

    # The tool result inject fails, then execution runs, then finally block fires.
    # Since inject keeps failing, the finally block's attempt also fails.
    # But the key point is process_tool_calls doesn't crash.
    await processor.process_tool_calls([tool_call])

    # The assistant message should be in history (first inject succeeded)
    assert len(context.conversation_history) >= 1
    assert context.conversation_history[0].role.value == "assistant"


@pytest.mark.asyncio
async def test_result_ids_reset_between_calls():
    """_result_ids_added is reset on each call to process_tool_calls."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    # First call
    tc1 = ToolCall(
        id="call_first",
        type="function",
        function=FunctionCall(name="tool_a", arguments="{}"),
    )
    await processor.process_tool_calls([tc1])
    assert "call_first" in processor._result_ids_added

    # Second call should start fresh
    tc2 = ToolCall(
        id="call_second",
        type="function",
        function=FunctionCall(name="tool_b", arguments="{}"),
    )
    await processor.process_tool_calls([tc2])
    assert "call_second" in processor._result_ids_added
    assert "call_first" not in processor._result_ids_added


@pytest.mark.asyncio
async def test_successful_batch_tracks_all_ids():
    """All tool_call_ids in a successful batch are tracked in _result_ids_added."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_calls = [
        ToolCall(
            id="batch_1",
            type="function",
            function=FunctionCall(name="tool_a", arguments="{}"),
        ),
        ToolCall(
            id="batch_2",
            type="function",
            function=FunctionCall(name="tool_b", arguments="{}"),
        ),
        ToolCall(
            id="batch_3",
            type="function",
            function=FunctionCall(name="tool_c", arguments="{}"),
        ),
    ]

    await processor.process_tool_calls(tool_calls)

    assert processor._result_ids_added == {"batch_1", "batch_2", "batch_3"}


# ---------------------------
# Tests for _truncate_tool_result
# ---------------------------


class TestTruncateToolResult:
    """Tests for ToolProcessor._truncate_tool_result()."""

    def _make_processor(self):
        tool_manager = DummyToolManager()
        context = DummyContext(tool_manager=tool_manager)
        ui_manager = DummyUIManager()
        return ToolProcessor(context, ui_manager)

    def test_under_limit_unchanged(self):
        processor = self._make_processor()
        content = "short content"
        result = processor._truncate_tool_result(content, max_chars=1000)
        assert result == content

    def test_exactly_at_limit_unchanged(self):
        processor = self._make_processor()
        content = "x" * 1000
        result = processor._truncate_tool_result(content, max_chars=1000)
        assert result == content

    def test_over_limit_truncated(self):
        processor = self._make_processor()
        content = "A" * 10_000
        result = processor._truncate_tool_result(content, max_chars=1000)
        assert len(result) < len(content)
        assert "TRUNCATED" in result
        assert "chars omitted" in result

    def test_preserves_head_and_tail(self):
        processor = self._make_processor()
        head = "HEAD_MARKER_" + "x" * 5000
        tail = "y" * 5000 + "_TAIL_MARKER"
        content = head + "z" * 90_000 + tail
        result = processor._truncate_tool_result(content, max_chars=10_000)
        assert result.startswith("HEAD_MARKER_")
        assert result.endswith("_TAIL_MARKER")

    def test_disabled_with_zero(self):
        processor = self._make_processor()
        content = "A" * 200_000
        result = processor._truncate_tool_result(content, max_chars=0)
        assert result == content

    def test_disabled_with_negative(self):
        processor = self._make_processor()
        content = "A" * 200_000
        result = processor._truncate_tool_result(content, max_chars=-1)
        assert result == content

    def test_empty_content(self):
        processor = self._make_processor()
        result = processor._truncate_tool_result("", max_chars=100)
        assert result == ""

    def test_value_binding_in_tail_preserved(self):
        """Value binding appended at end should survive truncation."""
        processor = self._make_processor()
        body = "A" * 200_000
        binding = "\n\n**RESULT: $v0 = 42.0**"
        content = body + binding
        result = processor._truncate_tool_result(content, max_chars=100_000)
        assert "**RESULT: $v0 = 42.0**" in result
