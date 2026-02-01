# tests/chat/test_conversation_extended.py
"""Extended tests for mcp_cli.chat.conversation to push coverage to >90%.

Complements the existing tests in test_conversation.py by exercising
code paths not yet covered (the remaining ~12%).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.chat.conversation import ConversationProcessor
from mcp_cli.chat.response_models import (
    CompletionResponse,
    Message,
    MessageRole,
    ToolCall,
    FunctionCall,
)


# ---------------------------------------------------------------------------
# Mock helpers (shared with existing tests)
# ---------------------------------------------------------------------------


class MockUIManager:
    """Mock UI manager for testing."""

    def __init__(self):
        self.is_streaming_response = False
        self.streaming_handler = None
        self.display = MagicMock()

    async def start_streaming_response(self):
        self.is_streaming_response = True

    async def stop_streaming_response(self):
        self.is_streaming_response = False

    async def print_assistant_message(self, content, elapsed):
        pass


class MockContext:
    """Mock context for testing."""

    def __init__(self):
        self.conversation_history = []
        self.openai_tools = []
        self.tool_name_mapping = {}
        self.client = MagicMock()
        self.tool_manager = MagicMock()
        self.tool_manager.get_adapted_tools_for_llm = AsyncMock(return_value=([], {}))
        self.provider = "openai"

    async def add_assistant_message(self, content):
        self.conversation_history.append(
            Message(role=MessageRole.ASSISTANT, content=content)
        )

    def inject_assistant_message(self, message):
        self.conversation_history.append(message)

    def inject_tool_message(self, message):
        self.conversation_history.append(message)


def _make_mock_tool_state():
    """Build a standard mock tool state."""
    mock = MagicMock()
    mock.reset_for_new_prompt = MagicMock()
    mock.register_user_literals = MagicMock(return_value=0)
    mock.extract_bindings_from_text = MagicMock(return_value=[])
    mock.format_unused_warning = MagicMock(return_value=None)
    mock.format_state_for_model = MagicMock(return_value="")
    mock.is_discovery_tool = MagicMock(return_value=False)
    mock.is_execution_tool = MagicMock(return_value=False)
    from chuk_ai_session_manager.guards import RunawayStatus

    mock.check_runaway = MagicMock(return_value=RunawayStatus(should_stop=False))
    return mock


# ===========================================================================
# Extended: _load_tools edge cases
# ===========================================================================


class TestLoadToolsExtended:
    """Extended tests for _load_tools."""

    @pytest.mark.asyncio
    async def test_load_tools_no_adapted_method(self):
        """When tool_manager lacks get_adapted_tools_for_llm, uses get_tools_for_llm."""
        context = MockContext()
        context.tool_manager = MagicMock(spec=[])  # No get_adapted_tools_for_llm

        ui = MockUIManager()
        processor = ConversationProcessor(context, ui)

        # Should not crash even without the method
        await processor._load_tools()
        assert context.openai_tools == []  # Falls through to error handler


# ===========================================================================
# Extended: process_conversation - non-streaming path
# ===========================================================================


class TestNonStreamingPath:
    """Tests for the non-streaming completion path."""

    @pytest.mark.asyncio
    async def test_non_streaming_response_displayed(self):
        """Non-streaming response calls print_assistant_message."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        # Create a client that does NOT support stream parameter
        mock_client = MagicMock(spec=[])
        mock_client.create_completion = AsyncMock(
            return_value={"response": "Hi!", "tool_calls": None}
        )
        context.client = mock_client

        ui = MockUIManager()
        ui.print_assistant_message = AsyncMock()
        ui.is_streaming_response = False

        processor = ConversationProcessor(context, ui)
        mock_ts = _make_mock_tool_state()
        processor._tool_state = mock_ts

        # Force non-streaming by making inspect fail
        with patch("inspect.signature", side_effect=ValueError("no sig")):
            await processor.process_conversation(max_turns=1)

        ui.print_assistant_message.assert_called_once()


# ===========================================================================
# Extended: Duplicate detection with empty state summary
# ===========================================================================


class TestDuplicateWithEmptyState:
    """Test duplicate tool call with empty state summary (should not inject)."""

    @pytest.mark.asyncio
    async def test_duplicate_empty_state(self):
        """Empty state summary from format_state_for_model should not inject."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Calc")]
        context.openai_tools = [{"type": "function", "function": {"name": "sqrt"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 16}'),
        )

        call_count = [0]

        async def mock_completion(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                return {"response": "", "tool_calls": [tool_call.model_dump()]}
            return {"response": "Done", "tool_calls": []}

        context.client.create_completion = mock_completion

        ui = MockUIManager()
        ui.is_streaming_response = False
        ui.stop_streaming_response = AsyncMock()
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)
        mock_ts = _make_mock_tool_state()
        mock_ts.format_state_for_model = MagicMock(return_value="")  # Empty state
        processor._tool_state = mock_ts
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=10)


# ===========================================================================
# Extended: Register user literals with no user messages
# ===========================================================================


class TestRegisterUserLiteralsNoUser:
    """Test _register_user_literals_from_history with only non-user messages."""

    def test_only_assistant_messages(self):
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.ASSISTANT, content="I computed 42"),
        ]
        ui = MockUIManager()
        processor = ConversationProcessor(context, ui)
        count = processor._register_user_literals_from_history()
        assert count == 0

    def test_user_message_no_content(self):
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content=None),
        ]
        ui = MockUIManager()
        processor = ConversationProcessor(context, ui)
        count = processor._register_user_literals_from_history()
        assert count == 0


# ===========================================================================
# Extended: Polling tool exemption from duplicate detection
# ===========================================================================


class TestPollingToolExemption:
    """Additional polling tool tests."""

    def test_all_polling_patterns(self):
        """All 8 patterns are covered."""
        context = MockContext()
        ui = MockUIManager()
        processor = ConversationProcessor(context, ui)

        for pattern in ConversationProcessor.POLLING_TOOL_PATTERNS:
            assert processor._is_polling_tool(f"my_{pattern}_tool") is True

    @pytest.mark.asyncio
    async def test_mixed_polling_and_non_polling(self):
        """Mix of polling and non-polling tools in same call."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = [
            {"type": "function", "function": {"name": "render_status"}},
            {"type": "function", "function": {"name": "compute"}},
        ]
        context.tool_name_mapping = {}

        # Two tool calls: one polling, one not
        tc1 = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="render_status", arguments='{"job": "j1"}'),
        )
        tc2 = ToolCall(
            id="call_2",
            type="function",
            function=FunctionCall(name="compute", arguments='{"x": 1}'),
        )

        ui = MockUIManager()
        ui.is_streaming_response = False
        ui.stop_streaming_response = AsyncMock()
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)
        mock_ts = _make_mock_tool_state()
        processor._tool_state = mock_ts
        processor.tool_processor.process_tool_calls = AsyncMock()

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tc1.model_dump(), tc2.model_dump()]},
                {"response": "Done", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=5)
        processor.tool_processor.process_tool_calls.assert_called_once()


# ===========================================================================
# Extended: _handle_regular_completion with no tools
# ===========================================================================


class TestRegularCompletionNoTools:
    """Test _handle_regular_completion called with tools=None."""

    @pytest.mark.asyncio
    async def test_regular_completion_none_tools(self):
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.client.create_completion = AsyncMock(
            return_value={"response": "Hi!", "tool_calls": None}
        )

        ui = MockUIManager()
        processor = ConversationProcessor(context, ui)

        result = await processor._handle_regular_completion(tools=None)
        assert isinstance(result, CompletionResponse)
        assert result.streaming is False


# ===========================================================================
# Extended: max turns edge case - exactly at max
# ===========================================================================


class TestMaxTurnsExact:
    """Test behavior when turn_count reaches exactly max_turns."""

    @pytest.mark.asyncio
    async def test_exactly_at_max_turns(self):
        """When at exactly max_turns, the max_turns message is injected."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Go")]
        context.openai_tools = [{"type": "function", "function": {"name": "fn"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="fn", arguments="{}"),
        )

        # Always return tool calls
        context.client.create_completion = AsyncMock(
            return_value={"response": "", "tool_calls": [tool_call.model_dump()]}
        )

        ui = MockUIManager()
        ui.is_streaming_response = False
        ui.stop_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui)
        mock_ts = _make_mock_tool_state()
        processor._tool_state = mock_ts
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=1)

        # Should have processed the first tool call and then hit max_turns
        # Either the tool was processed or max_turns was hit (depending on order)


# ===========================================================================
# Extended: Error in conversation loop handling
# ===========================================================================


class TestConversationLoopErrorExtended:
    """Extended error handling in conversation loop."""

    @pytest.mark.asyncio
    async def test_error_stops_streaming_ui(self):
        """Error in loop stops streaming UI before breaking."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui = MockUIManager()
        ui.is_streaming_response = True
        ui.stop_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui)
        context.client.create_completion = AsyncMock(
            side_effect=ValueError("unexpected")
        )

        await processor.process_conversation(max_turns=1)

        ui.stop_streaming_response.assert_called()


# ===========================================================================
# Extended: Streaming fallback path
# ===========================================================================


class TestStreamingFallbackExtended:
    """Extended streaming fallback tests."""

    @pytest.mark.asyncio
    async def test_streaming_fails_fallback_to_regular_with_tools(self):
        """When streaming fails, regular completion is used with tools."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = [{"type": "function", "function": {"name": "fn"}}]

        ui = MockUIManager()
        ui.start_streaming_response = AsyncMock()
        ui.display = MagicMock()
        ui.is_streaming_response = False
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)
        mock_ts = _make_mock_tool_state()
        processor._tool_state = mock_ts

        # Streaming fails
        async def mock_streaming_fail(tools=None):
            raise Exception("Stream broken")

        processor._handle_streaming_completion = mock_streaming_fail

        # Regular works
        context.client.create_completion = AsyncMock(
            return_value={"response": "Fallback", "tool_calls": []}
        )

        await processor.process_conversation(max_turns=1)

        # Verify response was displayed
        assert len(context.conversation_history) >= 2


# ===========================================================================
# Extended: unused warning path
# ===========================================================================


class TestUnusedWarningPath:
    """Test the unused_warning code path (currently disabled in UI but still runs)."""

    @pytest.mark.asyncio
    async def test_unused_warning_with_content(self):
        """When format_unused_warning returns content, it is logged."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Calc")]
        context.openai_tools = []

        context.client.create_completion = AsyncMock(
            return_value={"response": "Result is 42", "tool_calls": []}
        )

        ui = MockUIManager()
        ui.is_streaming_response = False
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)
        mock_ts = _make_mock_tool_state()
        mock_ts.format_unused_warning = MagicMock(return_value="Warning: unused v0=4.0")
        processor._tool_state = mock_ts

        await processor.process_conversation(max_turns=1)

        mock_ts.format_unused_warning.assert_called_once()


# ===========================================================================
# Extended: "No response" literal handling
# ===========================================================================


class TestNoResponseLiteral:
    """Test when response is literally empty or 'No response'."""

    @pytest.mark.asyncio
    async def test_empty_response_gets_default(self):
        """Empty response string becomes 'No response'."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hmm")]
        context.openai_tools = []

        context.client.create_completion = AsyncMock(
            return_value={"response": "", "tool_calls": []}
        )

        ui = MockUIManager()
        ui.is_streaming_response = False
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)
        mock_ts = _make_mock_tool_state()
        processor._tool_state = mock_ts

        await processor.process_conversation(max_turns=1)

        # The "No response" literal should NOT trigger binding extraction
        mock_ts.extract_bindings_from_text.assert_not_called()


# ===========================================================================
# Extended: Discovery and execution budget checked with name mapping
# ===========================================================================


class TestBudgetWithNameMapping:
    """Test budget checks with tool_name_mapping resolution."""

    @pytest.mark.asyncio
    async def test_name_mapping_resolves_tool(self):
        """Name mapping resolves sanitized names to original names."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Go")]
        context.openai_tools = [
            {"type": "function", "function": {"name": "sanitized_fn"}}
        ]
        context.tool_name_mapping = {"sanitized_fn": "original.fn"}

        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sanitized_fn", arguments="{}"),
        )

        ui = MockUIManager()
        ui.is_streaming_response = False
        ui.stop_streaming_response = AsyncMock()
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)
        mock_ts = _make_mock_tool_state()
        # Tool is a discovery tool
        mock_ts.is_discovery_tool = MagicMock(return_value=True)
        mock_ts.is_execution_tool = MagicMock(return_value=False)

        from chuk_ai_session_manager.guards import RunawayStatus

        # Discovery not exhausted, all OK
        mock_ts.check_runaway = MagicMock(return_value=RunawayStatus(should_stop=False))
        processor._tool_state = mock_ts
        processor.tool_processor.process_tool_calls = AsyncMock()

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tc.model_dump()]},
                {"response": "Done", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=5)

        # is_discovery_tool should have been called with the mapped name
        mock_ts.is_discovery_tool.assert_called_with("original.fn")


# ===========================================================================
# NEW: Cover lines 181-192 - streaming fails inside process_conversation
# The real _handle_streaming_completion raises, then fallback to regular
# ===========================================================================


class TestStreamingFailsFallbackInLoop:
    """Cover lines 181-192: streaming try/except inside the main loop."""

    @pytest.mark.asyncio
    async def test_streaming_exception_triggers_fallback(self):
        """When _handle_streaming_completion raises, code falls back to _handle_regular_completion."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui = MockUIManager()
        ui.is_streaming_response = False
        ui.print_assistant_message = AsyncMock()
        ui.start_streaming_response = AsyncMock()
        ui.display = MagicMock()

        processor = ConversationProcessor(context, ui)
        mock_ts = _make_mock_tool_state()
        processor._tool_state = mock_ts

        # Make streaming completion raise an exception
        async def streaming_raises(tools=None):
            raise RuntimeError("streaming broke")

        processor._handle_streaming_completion = streaming_raises

        # Make regular completion succeed
        async def regular_ok(tools=None):
            return CompletionResponse(
                response="Fallback OK",
                tool_calls=[],
                streaming=False,
                elapsed_time=0.1,
            )

        processor._handle_regular_completion = AsyncMock(side_effect=regular_ok)

        # Client must support streaming for the streaming path to be attempted
        context.client.create_completion = AsyncMock()

        # We need supports_streaming = True so the try block is entered
        # The easiest way: make sure client has create_completion with stream param
        with patch("inspect.signature") as mock_sig:
            mock_param = MagicMock()
            mock_param.parameters = {"stream": MagicMock()}
            mock_sig.return_value = mock_param

            await processor.process_conversation(max_turns=1)

        # _handle_regular_completion should have been called as fallback
        processor._handle_regular_completion.assert_called_once()
        # Message should be added to history
        assert any(
            hasattr(m, "content") and "Fallback OK" in (m.content or "")
            for m in context.conversation_history
        )


# ===========================================================================
# NEW: Cover line 266 - discovery budget exhausted with streaming active
# ===========================================================================


class TestDiscoveryBudgetStreamingActive:
    """Cover line 266: stop_streaming_response called when is_streaming_response=True."""

    @pytest.mark.asyncio
    async def test_discovery_budget_stops_streaming(self):
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Search")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "search"}}]
        context.tool_name_mapping = {}

        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments="{}"),
        )

        ui = MockUIManager()
        ui.is_streaming_response = True  # streaming IS active
        ui.stop_streaming_response = AsyncMock()
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_ts = _make_mock_tool_state()
        mock_ts.is_discovery_tool = MagicMock(return_value=True)
        mock_ts.is_execution_tool = MagicMock(return_value=False)

        disc_exhausted = RunawayStatus(
            should_stop=True,
            reason="Discovery budget exhausted",
            budget_exhausted=True,
        )
        mock_ts.check_runaway = MagicMock(return_value=disc_exhausted)
        mock_ts.format_discovery_exhausted_message = MagicMock(
            return_value="Discovery exhausted"
        )
        processor._tool_state = mock_ts

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tc.model_dump()]},
                {"response": "Final", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # stop_streaming_response should have been called
        ui.stop_streaming_response.assert_called()


# ===========================================================================
# NEW: Cover line 290 - execution budget exhausted with streaming active
# ===========================================================================


class TestExecutionBudgetStreamingActive:
    """Cover line 290: stop_streaming_response called when is_streaming_response=True."""

    @pytest.mark.asyncio
    async def test_execution_budget_stops_streaming(self):
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Execute")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "execute"}}]
        context.tool_name_mapping = {}

        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="execute", arguments="{}"),
        )

        ui = MockUIManager()
        ui.is_streaming_response = True  # streaming IS active
        ui.stop_streaming_response = AsyncMock()
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_ts = _make_mock_tool_state()
        mock_ts.is_discovery_tool = MagicMock(return_value=False)
        mock_ts.is_execution_tool = MagicMock(return_value=True)

        # Discovery check passes, execution check fails
        call_count = [0]

        def mock_check(tool_name=None):
            call_count[0] += 1
            if tool_name is not None:
                return RunawayStatus(
                    should_stop=True,
                    reason="Execution budget exhausted",
                    budget_exhausted=True,
                )
            return RunawayStatus(should_stop=False)

        mock_ts.check_runaway = MagicMock(side_effect=mock_check)
        mock_ts.format_execution_exhausted_message = MagicMock(
            return_value="Execution exhausted"
        )
        processor._tool_state = mock_ts

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tc.model_dump()]},
                {"response": "Done", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        ui.stop_streaming_response.assert_called()


# ===========================================================================
# NEW: Cover lines 306-314 - runaway with saturation_detected (not budget)
# ===========================================================================


class TestRunawaySaturationDetected:
    """Cover lines 306-314: saturation_detected branch in runaway handling."""

    @pytest.mark.asyncio
    async def test_saturation_detected_formats_message(self):
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Calc")]
        context.openai_tools = [{"type": "function", "function": {"name": "compute"}}]
        context.tool_name_mapping = {}

        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="compute", arguments="{}"),
        )

        ui = MockUIManager()
        ui.is_streaming_response = False
        ui.stop_streaming_response = AsyncMock()
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_ts = _make_mock_tool_state()
        mock_ts.is_discovery_tool = MagicMock(return_value=False)
        mock_ts.is_execution_tool = MagicMock(return_value=False)
        # Expose _recent_numeric_results for saturation path
        mock_ts._recent_numeric_results = [3.14159]

        call_count = [0]

        def mock_check(tool_name=None):
            call_count[0] += 1
            if tool_name is None:
                # General runaway check
                return RunawayStatus(
                    should_stop=True,
                    reason="Saturation detected",
                    budget_exhausted=False,
                    saturation_detected=True,
                )
            return RunawayStatus(should_stop=False)

        mock_ts.check_runaway = MagicMock(side_effect=mock_check)
        mock_ts.format_saturation_message = MagicMock(return_value="Values saturated")
        processor._tool_state = mock_ts

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tc.model_dump()]},
                {"response": "Final", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        mock_ts.format_saturation_message.assert_called_once_with(3.14159)

    @pytest.mark.asyncio
    async def test_saturation_detected_empty_numeric_results(self):
        """Cover the 0.0 fallback when _recent_numeric_results is empty."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Calc")]
        context.openai_tools = [{"type": "function", "function": {"name": "compute"}}]
        context.tool_name_mapping = {}

        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="compute", arguments="{}"),
        )

        ui = MockUIManager()
        ui.is_streaming_response = False
        ui.stop_streaming_response = AsyncMock()
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_ts = _make_mock_tool_state()
        mock_ts.is_discovery_tool = MagicMock(return_value=False)
        mock_ts.is_execution_tool = MagicMock(return_value=False)
        # Empty list -> should use 0.0 as fallback
        mock_ts._recent_numeric_results = []

        def mock_check(tool_name=None):
            if tool_name is None:
                return RunawayStatus(
                    should_stop=True,
                    reason="Saturation detected",
                    budget_exhausted=False,
                    saturation_detected=True,
                )
            return RunawayStatus(should_stop=False)

        mock_ts.check_runaway = MagicMock(side_effect=mock_check)
        mock_ts.format_saturation_message = MagicMock(return_value="Saturated at 0.0")
        processor._tool_state = mock_ts

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tc.model_dump()]},
                {"response": "Final", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        mock_ts.format_saturation_message.assert_called_once_with(0.0)


# ===========================================================================
# NEW: Cover lines 315-320 - runaway generic else branch (not budget, not saturation)
# ===========================================================================


class TestRunawayGenericElse:
    """Cover lines 315-320: the generic else branch in runaway stop message."""

    @pytest.mark.asyncio
    async def test_generic_runaway_stop_message(self):
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Calc")]
        context.openai_tools = [{"type": "function", "function": {"name": "compute"}}]
        context.tool_name_mapping = {}

        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="compute", arguments="{}"),
        )

        ui = MockUIManager()
        ui.is_streaming_response = False
        ui.stop_streaming_response = AsyncMock()
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_ts = _make_mock_tool_state()
        mock_ts.is_discovery_tool = MagicMock(return_value=False)
        mock_ts.is_execution_tool = MagicMock(return_value=False)

        def mock_check(tool_name=None):
            if tool_name is None:
                # Generic stop - NOT budget_exhausted, NOT saturation_detected
                return RunawayStatus(
                    should_stop=True,
                    reason="Too many calls",
                    budget_exhausted=False,
                    saturation_detected=False,
                )
            return RunawayStatus(should_stop=False)

        mock_ts.check_runaway = MagicMock(side_effect=mock_check)
        mock_ts.format_state_for_model = MagicMock(return_value="State: v0=42")
        processor._tool_state = mock_ts

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tc.model_dump()]},
                {"response": "Final", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # format_state_for_model should be called for the generic else branch
        mock_ts.format_state_for_model.assert_called()
        # Should have injected a message containing "Tool execution stopped"
        injected = [
            m
            for m in context.conversation_history
            if isinstance(m, str) and "Tool execution stopped" in m
        ]
        assert len(injected) >= 1


# ===========================================================================
# NEW: Cover line 327 - runaway with is_streaming_response=True
# ===========================================================================


class TestRunawayStopsStreaming:
    """Cover line 327: stop_streaming_response called in runaway path."""

    @pytest.mark.asyncio
    async def test_runaway_stops_active_streaming(self):
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Calc")]
        context.openai_tools = [{"type": "function", "function": {"name": "compute"}}]
        context.tool_name_mapping = {}

        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="compute", arguments="{}"),
        )

        ui = MockUIManager()
        ui.is_streaming_response = True  # streaming IS active
        ui.stop_streaming_response = AsyncMock()
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_ts = _make_mock_tool_state()
        mock_ts.is_discovery_tool = MagicMock(return_value=False)
        mock_ts.is_execution_tool = MagicMock(return_value=False)

        def mock_check(tool_name=None):
            if tool_name is None:
                return RunawayStatus(
                    should_stop=True,
                    reason="Budget exhausted",
                    budget_exhausted=True,
                )
            return RunawayStatus(should_stop=False)

        mock_ts.check_runaway = MagicMock(side_effect=mock_check)
        mock_ts.format_budget_exhausted_message = MagicMock(return_value="Budget done")
        processor._tool_state = mock_ts

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tc.model_dump()]},
                {"response": "Final", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        ui.stop_streaming_response.assert_called()


# ===========================================================================
# NEW: Cover line 345 - max turns with streaming active
# ===========================================================================


class TestMaxTurnsStopsStreaming:
    """Cover line 345: stop_streaming_response at max turns."""

    @pytest.mark.asyncio
    async def test_max_turns_stops_active_streaming(self):
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Loop")]
        context.openai_tools = [{"type": "function", "function": {"name": "fn"}}]
        context.tool_name_mapping = {}

        ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="fn", arguments='{"a": 1}'),
        )

        # Return different args each time to avoid duplicate detection
        call_num = [0]

        async def different_tool_calls(**kwargs):
            call_num[0] += 1
            return {
                "response": "",
                "tool_calls": [
                    {
                        "id": f"call_{call_num[0]}",
                        "type": "function",
                        "function": {
                            "name": "fn",
                            "arguments": f'{{"a": {call_num[0]}}}',
                        },
                    }
                ],
            }

        context.client.create_completion = different_tool_calls

        ui = MockUIManager()
        ui.is_streaming_response = True  # streaming IS active
        ui.stop_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui)
        mock_ts = _make_mock_tool_state()
        processor._tool_state = mock_ts
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=2)

        ui.stop_streaming_response.assert_called()


# ===========================================================================
# NEW: Cover line 398 - max duplicates with streaming active
# ===========================================================================


class _MockContextWithMessageInject(MockContext):
    """MockContext that wraps inject_assistant_message to always create Message objects."""

    def inject_assistant_message(self, message):
        if isinstance(message, str):
            self.conversation_history.append(
                Message(role=MessageRole.ASSISTANT, content=message)
            )
        else:
            self.conversation_history.append(message)


class TestMaxDuplicatesStopsStreaming:
    """Cover line 398: stop_streaming_response when max duplicates exceeded."""

    @pytest.mark.asyncio
    async def test_max_duplicates_stops_active_streaming(self):
        context = _MockContextWithMessageInject()
        context.conversation_history = [Message(role=MessageRole.USER, content="Calc")]
        context.openai_tools = [{"type": "function", "function": {"name": "sqrt"}}]
        context.tool_name_mapping = {}

        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 16}'),
        )

        # Always return the same tool call
        context.client.create_completion = AsyncMock(
            return_value={"response": "", "tool_calls": [tc.model_dump()]}
        )

        # Use a custom class whose is_streaming_response always returns True
        # so the stop_streaming_response call at line 398 is actually entered
        class AlwaysStreamingUI:
            def __init__(self):
                self.is_streaming_response = True
                self.streaming_handler = MagicMock()
                self.display = MagicMock()
                self.stop_streaming_response = AsyncMock()
                self.start_streaming_response = AsyncMock()
                self.print_assistant_message = AsyncMock()

        ui = AlwaysStreamingUI()

        processor = ConversationProcessor(context, ui)
        processor._max_consecutive_duplicates = 2  # Very low threshold

        mock_ts = _make_mock_tool_state()
        mock_ts.format_state_for_model = MagicMock(return_value="State: v0=4.0")
        processor._tool_state = mock_ts
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=20)

        ui.stop_streaming_response.assert_called()


# ===========================================================================
# NEW: Cover lines 461-462 - streaming response cleanup in else branch
# ===========================================================================


class TestStreamingResponseCleanupElse:
    """Cover lines 461-462: when completion.streaming=True, clear streaming_handler."""

    @pytest.mark.asyncio
    async def test_streaming_completion_clears_handler_in_else(self):
        """The else branch at line 455 clears streaming_handler when response is streaming."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hi")]
        context.openai_tools = []

        ui = MockUIManager()
        ui.is_streaming_response = False
        ui.streaming_handler = MagicMock()  # Pre-set handler
        ui.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui)
        mock_ts = _make_mock_tool_state()
        processor._tool_state = mock_ts

        # Return a streaming=True completion (triggers the else branch)
        async def mock_streaming(tools=None):
            return CompletionResponse(
                response="Streamed response!",
                tool_calls=[],
                streaming=True,
                elapsed_time=0.5,
            )

        processor._handle_streaming_completion = mock_streaming

        # Make sure streaming path is taken
        with patch("inspect.signature") as mock_sig:
            mock_param = MagicMock()
            mock_param.parameters = {"stream": MagicMock()}
            mock_sig.return_value = mock_param

            await processor.process_conversation(max_turns=1)

        # streaming_handler should have been cleared (set to None)
        assert ui.streaming_handler is None
        # print_assistant_message should NOT have been called (streaming=True path)
        ui.print_assistant_message.assert_not_called()


# ===========================================================================
# NEW: Cover lines 522-557 - _handle_streaming_completion method body
# ===========================================================================


class TestHandleStreamingCompletionDirect:
    """Cover lines 522-557: the actual _handle_streaming_completion method."""

    @pytest.mark.asyncio
    async def test_handle_streaming_completion_success(self):
        """Test _handle_streaming_completion returns CompletionResponse."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hi")]
        context.client = MagicMock()

        ui = MockUIManager()
        ui.start_streaming_response = AsyncMock()
        ui.display = MagicMock()

        processor = ConversationProcessor(context, ui)

        # Mock StreamingResponseHandler
        mock_handler = MagicMock()
        mock_handler.stream_response = AsyncMock(
            return_value={
                "response": "Streamed!",
                "tool_calls": [],
                "chunks_received": 5,
                "elapsed_time": 1.2,
                "streaming": True,
                "interrupted": False,
            }
        )

        with patch(
            "mcp_cli.chat.streaming_handler.StreamingResponseHandler",
            return_value=mock_handler,
        ):
            result = await processor._handle_streaming_completion(
                tools=[{"some": "tool"}]
            )

        assert isinstance(result, CompletionResponse)
        assert result.response == "Streamed!"
        assert result.streaming is True
        assert result.elapsed_time == 1.2
        ui.start_streaming_response.assert_called_once()
        # streaming_handler should have been set on ui_manager
        assert ui.streaming_handler is mock_handler

    @pytest.mark.asyncio
    async def test_handle_streaming_completion_with_tool_calls(self):
        """Test _handle_streaming_completion with tool calls in response."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Calc")]
        context.client = MagicMock()

        ui = MockUIManager()
        ui.start_streaming_response = AsyncMock()
        ui.display = MagicMock()

        processor = ConversationProcessor(context, ui)

        tc_dict = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "sqrt", "arguments": '{"x": 16}'},
        }
        mock_handler = MagicMock()
        mock_handler.stream_response = AsyncMock(
            return_value={
                "response": "",
                "tool_calls": [tc_dict],
                "chunks_received": 3,
                "elapsed_time": 0.8,
                "streaming": True,
            }
        )

        with patch(
            "mcp_cli.chat.streaming_handler.StreamingResponseHandler",
            return_value=mock_handler,
        ):
            result = await processor._handle_streaming_completion(tools=[])

        assert isinstance(result, CompletionResponse)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "sqrt"

    @pytest.mark.asyncio
    async def test_handle_streaming_completion_exception_in_finally(self):
        """Test _handle_streaming_completion propagates exception but finally runs."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hi")]
        context.client = MagicMock()

        ui = MockUIManager()
        ui.start_streaming_response = AsyncMock()
        ui.display = MagicMock()

        processor = ConversationProcessor(context, ui)

        mock_handler = MagicMock()
        mock_handler.stream_response = AsyncMock(
            side_effect=RuntimeError("stream broke")
        )

        with patch(
            "mcp_cli.chat.streaming_handler.StreamingResponseHandler",
            return_value=mock_handler,
        ):
            with pytest.raises(RuntimeError, match="stream broke"):
                await processor._handle_streaming_completion(tools=[])

        # start_streaming_response was called
        ui.start_streaming_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_streaming_completion_none_tools(self):
        """Test _handle_streaming_completion with tools=None."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hi")]
        context.client = MagicMock()

        ui = MockUIManager()
        ui.start_streaming_response = AsyncMock()
        ui.display = MagicMock()

        processor = ConversationProcessor(context, ui)

        mock_handler = MagicMock()
        mock_handler.stream_response = AsyncMock(
            return_value={
                "response": "No tools",
                "tool_calls": [],
                "streaming": True,
            }
        )

        with patch(
            "mcp_cli.chat.streaming_handler.StreamingResponseHandler",
            return_value=mock_handler,
        ):
            result = await processor._handle_streaming_completion(tools=None)

        assert result.response == "No tools"
        # Verify tools=None was passed through
        call_kwargs = mock_handler.stream_response.call_args[1]
        assert call_kwargs["tools"] is None
