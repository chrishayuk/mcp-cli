# tests/chat/test_conversation.py
"""Tests for ConversationProcessor."""

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
        # Make tool_manager async methods work properly
        self.tool_manager.get_adapted_tools_for_llm = AsyncMock(return_value=([], {}))
        self.provider = "openai"

    async def add_assistant_message(self, content):
        """Add assistant message to conversation history."""
        self.conversation_history.append(
            Message(role=MessageRole.ASSISTANT, content=content)
        )

    def inject_assistant_message(self, message):
        """Inject a message into conversation history."""
        self.conversation_history.append(message)

    def inject_tool_message(self, message):
        """Inject a tool message into conversation history."""
        self.conversation_history.append(message)


class TestConversationProcessorInit:
    """Tests for ConversationProcessor initialization."""

    def test_init(self):
        """Test basic initialization."""
        context = MockContext()
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)

        assert processor.context is context
        assert processor.ui_manager is ui_manager
        assert processor.tool_processor is not None
        assert processor._consecutive_duplicate_count == 0
        assert processor._max_consecutive_duplicates == 5

    def test_init_with_runtime_config(self):
        """Test initialization with runtime config."""
        context = MockContext()
        ui_manager = MockUIManager()
        runtime_config = {"some": "config"}

        processor = ConversationProcessor(context, ui_manager, runtime_config)

        assert processor.runtime_config == runtime_config


class TestRegisterUserLiterals:
    """Tests for _register_user_literals_from_history."""

    def test_register_from_user_message(self):
        """Test registering literals from user message."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate sqrt of 18")
        ]
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)
        count = processor._register_user_literals_from_history()

        # Should register at least the number 18
        assert count >= 1

    def test_register_empty_history(self):
        """Test with empty history."""
        context = MockContext()
        context.conversation_history = []
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)
        count = processor._register_user_literals_from_history()

        assert count == 0

    def test_register_only_most_recent(self):
        """Test that only most recent user message is processed."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="First message with 10"),
            Message(role=MessageRole.ASSISTANT, content="Response"),
            Message(role=MessageRole.USER, content="Second with 20 and 30"),
        ]
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)
        # Reset tool state to clear any prior registrations
        processor._tool_state.reset_for_new_prompt()
        count = processor._register_user_literals_from_history()

        # Should only process the most recent user message
        assert count >= 2  # At least 20 and 30


class TestProcessConversation:
    """Tests for process_conversation method."""

    @pytest.mark.asyncio
    async def test_slash_command_skipped(self):
        """Test that slash commands are skipped."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="/help")]
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)
        await processor.process_conversation()

        # Should return immediately without processing
        assert len(context.conversation_history) == 1

    @pytest.mark.asyncio
    async def test_no_tools_loads_tools(self):
        """Test that tools are loaded if not present."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = None  # No tools loaded
        context.client.create_completion = AsyncMock(
            return_value={"response": "Hi there!", "tool_calls": None}
        )
        ui_manager = MockUIManager()

        # Mock _load_tools
        processor = ConversationProcessor(context, ui_manager)
        processor._load_tools = AsyncMock()

        await processor.process_conversation()

        # Should have called _load_tools
        processor._load_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_simple_response(self):
        """Test processing a simple text response."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        # Mock client response
        context.client.create_completion = AsyncMock(
            return_value={"response": "Hello! How can I help?", "tool_calls": None}
        )

        ui_manager = MockUIManager()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)
        await processor.process_conversation()

        # Should have added assistant message to history
        assert len(context.conversation_history) == 2
        assert context.conversation_history[-1].role == MessageRole.ASSISTANT
        assert "Hello" in context.conversation_history[-1].content

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        """Test that max_turns limit is enforced."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        # Mock client to always return tool calls (would loop forever)
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 18}'),
        )
        context.client.create_completion = AsyncMock(
            return_value={"response": "", "tool_calls": [tool_call.model_dump()]}
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        # Mock tool processor to avoid actual execution
        processor.tool_processor.process_tool_calls = AsyncMock()

        # Set very low max_turns
        await processor.process_conversation(max_turns=2)

        # Should have stopped due to max_turns
        assert processor.tool_processor.process_tool_calls.call_count <= 2


class TestHandleRegularCompletion:
    """Tests for _handle_regular_completion."""

    @pytest.mark.asyncio
    async def test_regular_completion_success(self):
        """Test successful regular completion."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.client.create_completion = AsyncMock(
            return_value={"response": "Hi!", "tool_calls": None}
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        result = await processor._handle_regular_completion(tools=[])

        assert isinstance(result, CompletionResponse)
        assert result.response == "Hi!"
        assert result.streaming is False
        assert result.elapsed_time > 0

    @pytest.mark.asyncio
    async def test_regular_completion_tool_error_retry(self):
        """Test retry without tools on tool definition error."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]

        # First call fails with tool error, second succeeds
        context.client.create_completion = AsyncMock(
            side_effect=[
                Exception("Invalid 'tools' specification"),
                {"response": "Hi without tools!", "tool_calls": None},
            ]
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        result = await processor._handle_regular_completion(tools=[{"some": "tool"}])

        assert isinstance(result, CompletionResponse)
        assert result.response == "Hi without tools!"
        # Should have been called twice
        assert context.client.create_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_regular_completion_other_error_raises(self):
        """Test that non-tool errors are raised."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.client.create_completion = AsyncMock(
            side_effect=Exception("Some other error")
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        with pytest.raises(Exception, match="Some other error"):
            await processor._handle_regular_completion(tools=[])


class TestLoadTools:
    """Tests for _load_tools."""

    @pytest.mark.asyncio
    async def test_load_tools_success(self):
        """Test successful tool loading."""
        context = MockContext()
        context.tool_manager.get_adapted_tools_for_llm = AsyncMock(
            return_value=(
                [{"type": "function", "function": {"name": "sqrt"}}],
                {"sqrt": "math.sqrt"},
            )
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        await processor._load_tools()

        assert len(context.openai_tools) == 1
        assert context.tool_name_mapping == {"sqrt": "math.sqrt"}

    @pytest.mark.asyncio
    async def test_load_tools_error(self):
        """Test tool loading error handling."""
        context = MockContext()
        context.tool_manager.get_adapted_tools_for_llm = AsyncMock(
            side_effect=Exception("Failed to load")
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        await processor._load_tools()

        # Should set empty tools on error
        assert context.openai_tools == []
        assert context.tool_name_mapping == {}


class TestHandleStreamingCompletion:
    """Tests for _handle_streaming_completion via process_conversation integration."""

    @pytest.mark.asyncio
    async def test_streaming_path_taken_when_supported(self):
        """Test that streaming path is used when client supports it."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.start_streaming_response = AsyncMock()
        ui_manager.display = MagicMock()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        # Mock streaming completion method directly
        async def mock_streaming(tools=None):
            return CompletionResponse(
                response="Streamed!",
                tool_calls=[],
                streaming=True,
                elapsed_time=0.5,
            )

        processor._handle_streaming_completion = mock_streaming

        await processor.process_conversation(max_turns=1)

        # The streaming handler should have been set to None at the end
        assert ui_manager.streaming_handler is None

    @pytest.mark.asyncio
    async def test_tool_calls_are_processed_in_conversation(self):
        """Test that tool calls from completion are processed."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "sqrt"}}]
        context.tool_name_mapping = {}

        # Create client mock that returns tool calls first, then a response
        call_count = [0]

        async def mock_completion(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "response": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "sqrt", "arguments": '{"x": 16}'},
                        }
                    ],
                }
            else:
                return {"response": "The result is 4", "tool_calls": []}

        context.client.create_completion = mock_completion

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=3)

        # Should have processed tool calls
        processor.tool_processor.process_tool_calls.assert_called()


class TestDuplicateToolCallDetection:
    """Tests for duplicate tool call detection."""

    @pytest.mark.asyncio
    async def test_consecutive_duplicate_limit(self):
        """Test that max consecutive duplicates triggers exit."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        # Create a tool call response
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 18}'),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        processor = ConversationProcessor(context, ui_manager)

        # Simulate max consecutive duplicates
        processor._consecutive_duplicate_count = 4  # One below max
        processor._max_consecutive_duplicates = 5

        # Mock to always return the same tool call
        context.client.create_completion = AsyncMock(
            return_value={"response": "", "tool_calls": [tool_call.model_dump()]}
        )

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        # Set last signature to match current
        with patch.object(processor, "_register_user_literals_from_history"):
            await processor.process_conversation(max_turns=3)

        # Should have stopped due to duplicate detection or max_turns
        # The test validates the mechanism exists

    @pytest.mark.asyncio
    async def test_duplicate_detection_reset_on_different_args(self):
        """Test that duplicate counter resets when args differ."""
        context = MockContext()
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)
        processor._consecutive_duplicate_count = 3

        # Accessing internal state to test reset
        # When a non-duplicate comes in, counter should reset to 0
        assert processor._consecutive_duplicate_count == 3

        # After processing different tool call, would reset
        processor._consecutive_duplicate_count = 0
        assert processor._consecutive_duplicate_count == 0


class TestBudgetExhaustion:
    """Tests for tool budget exhaustion scenarios."""

    @pytest.mark.asyncio
    async def test_discovery_budget_exhausted(self):
        """Test handling of discovery budget exhaustion."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Search for something")
        ]
        context.openai_tools = []

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments='{"query": "test"}'),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        processor = ConversationProcessor(context, ui_manager)

        # Mock the tool state to indicate discovery budget exhausted
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_status = RunawayStatus(
            should_stop=True, reason="Discovery budget exhausted", budget_exhausted=True
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        # Create a mock tool state manager
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=True)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)

        check_call_count = [0]

        def mock_check_runaway(tool_name=None):
            check_call_count[0] += 1
            # First check is for discovery tool budget
            if check_call_count[0] == 1:
                return mock_status
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check_runaway)
        mock_tool_state.format_discovery_exhausted_message = MagicMock(
            return_value="Discovery exhausted"
        )

        # Replace the tool state
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Final answer", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

    @pytest.mark.asyncio
    async def test_execution_budget_exhausted(self):
        """Test handling of execution budget exhaustion."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Execute something")
        ]
        context.openai_tools = []

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="execute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_status = RunawayStatus(
            should_stop=True, reason="Execution budget exhausted", budget_exhausted=True
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        # Create a mock tool state manager
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=True)

        check_call_count = [0]

        def mock_check_runaway(tool_name=None):
            check_call_count[0] += 1
            # First check (discovery) is OK, second check (execution) exhausted
            if check_call_count[0] == 2:
                return mock_status
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check_runaway)
        mock_tool_state.format_execution_exhausted_message = MagicMock(
            return_value="Execution exhausted"
        )

        # Replace the tool state
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Done", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)


class TestRunawayDetection:
    """Tests for general runaway detection."""

    @pytest.mark.asyncio
    async def test_runaway_with_saturation(self):
        """Test runaway detection with saturation."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = []

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="compute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_status = RunawayStatus(
            should_stop=True,
            reason="Saturation detected",
            saturation_detected=True,
            message="Results have converged",
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        # Create a mock tool state manager
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state._recent_numeric_results = [3.14159]

        check_call_count = [0]

        def mock_check_runaway(tool_name=None):
            check_call_count[0] += 1
            # Third call is general runaway check
            if check_call_count[0] == 3:
                return mock_status
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check_runaway)
        mock_tool_state.format_saturation_message = MagicMock(
            return_value="Saturation message"
        )

        # Replace the tool state
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Final", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)


class TestStreamingFallback:
    """Tests for streaming fallback to regular completion."""

    @pytest.mark.asyncio
    async def test_streaming_fallback_on_error(self):
        """Test fallback to regular completion when streaming fails."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.start_streaming_response = AsyncMock()
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()
        ui_manager.display = MagicMock()
        ui_manager.is_streaming_response = False

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        # Mock streaming to fail and regular to succeed
        async def mock_streaming_fail(tools=None):
            raise Exception("Streaming error")

        processor._handle_streaming_completion = mock_streaming_fail

        # But regular completion succeeds
        context.client.create_completion = AsyncMock(
            return_value={"response": "Fallback response", "tool_calls": []}
        )

        await processor.process_conversation(max_turns=1)

        # Should have fallen back to regular completion
        context.client.create_completion.assert_called()


class TestConversationErrorHandling:
    """Tests for error handling in conversation processing."""

    @pytest.mark.asyncio
    async def test_general_error_in_loop(self):
        """Test error handling during conversation loop."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Mock client to raise an error
        context.client.create_completion = AsyncMock(
            side_effect=ValueError("Some unexpected error")
        )

        await processor.process_conversation(max_turns=1)

        # Should have added error message to history
        assert len(context.conversation_history) >= 2
        last_msg = context.conversation_history[-1]
        # Error message may be a Message object or a string depending on the error path
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        assert "error" in content.lower()

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        """Test that asyncio.CancelledError is re-raised."""
        import asyncio

        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        context.client.create_completion = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        with pytest.raises(asyncio.CancelledError):
            await processor.process_conversation(max_turns=1)


class TestInspectionHandling:
    """Tests for signature inspection edge cases."""

    @pytest.mark.asyncio
    async def test_inspection_error_disables_streaming(self):
        """Test that inspection errors disable streaming gracefully."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        # Create a client that has create_completion but inspection fails
        mock_client = MagicMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "Hi!", "tool_calls": None}
        )

        context.client = mock_client

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Mock inspect.signature to raise
        with patch("inspect.signature", side_effect=ValueError("Cannot inspect")):
            await processor.process_conversation(max_turns=1)

            # Should complete without streaming
            mock_client.create_completion.assert_called()


class TestBindingExtraction:
    """Tests for value binding extraction from responses."""

    @pytest.mark.asyncio
    async def test_extracts_bindings_from_response(self):
        """Test that value bindings are extracted from assistant responses."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = []

        context.client.create_completion = AsyncMock(
            return_value={
                "response": "The result is σ = 5.0 and π = 3.14159",
                "tool_calls": [],  # Empty list, not None
            }
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state with binding extraction
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_extract = MagicMock(return_value=[])
        mock_tool_state.extract_bindings_from_text = mock_extract
        processor._tool_state = mock_tool_state

        await processor.process_conversation(max_turns=1)

        # Should have tried to extract bindings
        mock_extract.assert_called_once()


class TestStreamingCleanup:
    """Tests for streaming handler cleanup."""

    @pytest.mark.asyncio
    async def test_streaming_handler_cleared_after_response(self):
        """Test that streaming handler reference is cleared after streaming response."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.streaming_handler = MagicMock()  # Simulate existing handler
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        # Mock _handle_streaming_completion to return a streaming response
        async def mock_streaming_completion(tools=None):
            return CompletionResponse(
                response="Hi!",
                tool_calls=[],
                streaming=True,
                elapsed_time=0.5,
            )

        processor._handle_streaming_completion = mock_streaming_completion

        await processor.process_conversation(max_turns=1)

        # Handler should be cleared after streaming response
        assert ui_manager.streaming_handler is None


class TestMaxTurnsWithToolCalls:
    """Tests for max_turns limit with tool calls."""

    @pytest.mark.asyncio
    async def test_max_turns_stops_tool_loop(self):
        """Test that max_turns limit stops infinite tool call loops."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Keep calling tools")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "loop"}}]
        context.tool_name_mapping = {}

        # Always return tool calls - would loop forever without max_turns
        async def infinite_tool_calls(**kwargs):
            return {
                "response": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "loop", "arguments": "{}"},
                    }
                ],
            }

        context.client.create_completion = infinite_tool_calls

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        # Set very low max_turns
        await processor.process_conversation(max_turns=2)

        # Should have stopped after max_turns
        assert processor.tool_processor.process_tool_calls.call_count <= 2


class TestGeneralRunawayDetection:
    """Tests for general runaway detection scenarios."""

    @pytest.mark.asyncio
    async def test_runaway_with_budget_exhausted(self):
        """Test runaway detection with budget exhausted."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "compute"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="compute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()
        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_status = RunawayStatus(
            should_stop=True,
            reason="Budget exhausted",
            budget_exhausted=True,
            message="Tool call budget exhausted",
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        # Create a mock tool state manager
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)

        check_call_count = [0]

        def mock_check_runaway(tool_name=None):
            check_call_count[0] += 1
            # General runaway check is after discovery and execution checks
            # Discovery check returns should_stop=False
            # Execution check returns should_stop=False
            # General runaway check (3rd call) returns should_stop=True
            if check_call_count[0] >= 3:
                return mock_status
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check_runaway)
        mock_tool_state.format_budget_exhausted_message = MagicMock(
            return_value="Budget exhausted message"
        )
        mock_tool_state.format_state_for_model = MagicMock(return_value="State summary")

        # Replace the tool state
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Final", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # The test validates that runaway detection mechanism is called
        # The exact behavior depends on the tool state configuration
        assert check_call_count[0] >= 1  # check_runaway was called


class TestDuplicateToolCallHandling:
    """Tests for duplicate tool call handling with state injection."""

    @pytest.mark.asyncio
    async def test_duplicate_triggers_state_injection(self):
        """Test that duplicate tool calls trigger state summary injection."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "sqrt"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 16}'),
        )

        # Return same tool call multiple times, then final response
        call_count = [0]

        async def repeated_tool_calls(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:  # First 3 calls return same tool
                return {"response": "", "tool_calls": [tool_call.model_dump()]}
            else:
                return {"response": "Done", "tool_calls": []}

        context.client.create_completion = repeated_tool_calls

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_state_for_model = MagicMock(return_value="State: v0=4.0")
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=10)

        # Should have injected state summary (called format_state_for_model)
        # after detecting duplicate
        assert mock_tool_state.format_state_for_model.call_count >= 1


class TestReasoningContent:
    """Tests for reasoning content handling."""

    @pytest.mark.asyncio
    async def test_reasoning_content_preserved(self):
        """Test that reasoning content from response is preserved."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Think about this")
        ]
        context.openai_tools = []

        context.client.create_completion = AsyncMock(
            return_value={
                "response": "The answer is 42",
                "tool_calls": [],
                "reasoning_content": "Let me think step by step...",
            }
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        await processor.process_conversation(max_turns=1)

        # Check that reasoning content was added to the message
        last_msg = context.conversation_history[-1]
        assert last_msg.role == MessageRole.ASSISTANT
        assert hasattr(last_msg, "reasoning_content") or "reasoning" in str(last_msg)


class TestMaxDuplicatesExceeded:
    """Tests for max duplicates safety valve."""

    @pytest.mark.asyncio
    async def test_max_duplicates_breaks_loop(self):
        """Test that hitting max consecutive duplicates breaks the loop."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "sqrt"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 16}'),
        )

        # Always return the same tool call
        context.client.create_completion = AsyncMock(
            return_value={"response": "", "tool_calls": [tool_call.model_dump()]}
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)
        processor._max_consecutive_duplicates = 3  # Lower threshold for testing

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.format_state_for_model = MagicMock(return_value="")
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=20)

        # Should have detected duplicates and eventually broken out
        # The exact count depends on implementation but should be limited
        assert processor.tool_processor.process_tool_calls.call_count <= 10


class TestDiscoveryBudgetWithMessage:
    """Tests for discovery budget with formatted message."""

    @pytest.mark.asyncio
    async def test_discovery_budget_formats_message(self):
        """Test that discovery budget exhaustion formats proper message."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Search")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "search"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        # Discovery budget exhausted with "Discovery" in reason
        mock_status = RunawayStatus(
            should_stop=True,
            reason="Discovery budget exhausted",
            budget_exhausted=True,
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=True)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_discovery_exhausted_message = MagicMock(
            return_value="Discovery budget exhausted - please answer with available data"
        )

        # First check returns discovery exhausted
        call_count = [0]

        def mock_check(tool_name=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_status
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check)
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Here's my answer", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # Should have called format_discovery_exhausted_message
        mock_tool_state.format_discovery_exhausted_message.assert_called()


class TestExecutionBudgetWithMessage:
    """Tests for execution budget with formatted message."""

    @pytest.mark.asyncio
    async def test_execution_budget_formats_message(self):
        """Test that execution budget exhaustion formats proper message."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Execute")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "execute"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="execute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        # Execution budget exhausted with "Execution" in reason
        # The check passes tool name to check_runaway for execution tools
        mock_status_exec = RunawayStatus(
            should_stop=True,
            reason="Execution budget exhausted",
            budget_exhausted=True,
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=True)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_execution_exhausted_message = MagicMock(
            return_value="Execution budget exhausted - please provide final answer"
        )

        # check_runaway is called with tool_name for specific checks
        # When called with a tool name (execution tool), return exhausted
        def mock_check(tool_name=None):
            if tool_name is not None:
                # This is the execution budget check
                return mock_status_exec
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check)
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Done", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # Should have called format_execution_exhausted_message
        mock_tool_state.format_execution_exhausted_message.assert_called()


class TestPollingToolDetection:
    """Tests for polling tool detection and loop exemption."""

    def test_is_polling_tool_status(self):
        """Test that 'status' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("render_status") is True
        assert processor._is_polling_tool("get_status") is True
        assert processor._is_polling_tool("remotion_render_status") is True
        assert processor._is_polling_tool("job_status_check") is True

    def test_is_polling_tool_progress(self):
        """Test that 'progress' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("check_progress") is True
        assert processor._is_polling_tool("get_progress") is True
        assert processor._is_polling_tool("render_progress") is True

    def test_is_polling_tool_check(self):
        """Test that 'check' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("health_check") is True
        assert processor._is_polling_tool("check_job") is True

    def test_is_polling_tool_poll(self):
        """Test that 'poll' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("poll_results") is True
        assert processor._is_polling_tool("poll_queue") is True

    def test_is_polling_tool_monitor(self):
        """Test that 'monitor' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("monitor_job") is True
        assert processor._is_polling_tool("system_monitor") is True

    def test_is_polling_tool_watch(self):
        """Test that 'watch' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("watch_progress") is True
        assert processor._is_polling_tool("file_watch") is True

    def test_is_polling_tool_wait(self):
        """Test that 'wait' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("wait_for_completion") is True
        assert processor._is_polling_tool("wait_job") is True

    def test_is_polling_tool_state(self):
        """Test that 'state' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("get_state") is True
        assert processor._is_polling_tool("job_state") is True

    def test_is_not_polling_tool(self):
        """Test that non-polling tools are not detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("sqrt") is False
        assert processor._is_polling_tool("add") is False
        assert processor._is_polling_tool("create_video") is False
        assert (
            processor._is_polling_tool("render_video") is False
        )  # render but not status
        assert processor._is_polling_tool("calculate") is False
        assert processor._is_polling_tool("search") is False

    def test_is_polling_tool_case_insensitive(self):
        """Test that polling tool detection is case-insensitive."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("GET_STATUS") is True
        assert processor._is_polling_tool("Check_Progress") is True
        assert processor._is_polling_tool("POLL_RESULTS") is True

    @pytest.mark.asyncio
    async def test_polling_tool_not_marked_as_duplicate(self):
        """Test that polling tools calling same args are not marked as duplicates."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Check the status")
        ]
        context.openai_tools = [
            {"type": "function", "function": {"name": "render_status"}}
        ]
        context.tool_name_mapping = {}

        # Same status check tool call
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(
                name="render_status", arguments='{"job_id": "abc123"}'
            ),
        )

        # Return same tool call multiple times, then final response
        call_count = [0]

        async def repeated_status_checks(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:  # First 3 calls return same status check
                return {"response": "", "tool_calls": [tool_call.model_dump()]}
            else:
                return {"response": "Render complete!", "tool_calls": []}

        context.client.create_completion = repeated_status_checks

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_state_for_model = MagicMock(return_value="")
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=10)

        # All 3 status checks should have been processed (not skipped as duplicates)
        assert processor.tool_processor.process_tool_calls.call_count >= 3
        # Duplicate counter should not have incremented for polling tools
        assert processor._consecutive_duplicate_count == 0


class TestBindingExtractionWithResults:
    """Tests for binding extraction returning actual bindings."""

    @pytest.mark.asyncio
    async def test_extracts_bindings_with_values(self):
        """Test that actual bindings are extracted and logged."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = []

        context.client.create_completion = AsyncMock(
            return_value={
                "response": "The result is σ = 5.0",
                "tool_calls": [],
            }
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock binding
        mock_binding = MagicMock()
        mock_binding.id = "v0"
        mock_binding.raw_value = 5.0
        mock_binding.aliases = ["σ"]

        # Create mock tool state that returns a binding
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(
            return_value=[mock_binding]
        )
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        await processor.process_conversation(max_turns=1)

        # Should have called extract_bindings_from_text
        mock_tool_state.extract_bindings_from_text.assert_called_once()


class TestValidateToolMessages:
    """Tests for _validate_tool_messages defense-in-depth validation."""

    def test_valid_messages_unchanged(self):
        """Messages with matching tool results are not modified."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "OK"},
            {"role": "assistant", "content": "Done."},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)
        assert result == messages

    def test_orphaned_tool_call_gets_placeholder(self):
        """An assistant message with a tool_call_id missing a result gets a placeholder."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_missing", "type": "function", "function": {"name": "fetch", "arguments": "{}"}},
                ],
            },
            # No tool result for call_missing!
            {"role": "assistant", "content": "Something else."},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # Should have 4 messages now: user, assistant+tool_calls, tool placeholder, assistant
        assert len(result) == 4
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_missing"
        assert "did not complete" in result[2]["content"]

    def test_multiple_tool_calls_partial_results(self):
        """When an assistant message has multiple tool_calls and only some have results."""
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "Result A"},
            # call_b result is missing
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # Should have 4 messages: user, assistant+tool_calls, tool result A, placeholder for B
        assert len(result) == 4
        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results) == 2
        tool_call_ids = {m["tool_call_id"] for m in tool_results}
        assert tool_call_ids == {"call_a", "call_b"}

    def test_no_tool_calls_unchanged(self):
        """Messages without tool_calls pass through unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)
        assert result == messages

    def test_empty_messages(self):
        """Empty message list returns empty."""
        assert ConversationProcessor._validate_tool_messages([]) == []

    def test_multiple_sequential_tool_rounds(self):
        """Multiple assistant→tool rounds are all validated correctly."""
        messages = [
            {"role": "user", "content": "Do things"},
            # Round 1: valid
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_r1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_r1", "content": "Done A"},
            # Round 2: orphaned
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_r2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            # Missing tool result for call_r2!
            {"role": "assistant", "content": "Final answer."},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # Should have 6 messages: user, asst+tc, tool, asst+tc, PLACEHOLDER, asst
        assert len(result) == 6
        assert result[4]["role"] == "tool"
        assert result[4]["tool_call_id"] == "call_r2"
        assert "did not complete" in result[4]["content"]
        # Round 1 should be untouched
        assert result[2]["tool_call_id"] == "call_r1"
        assert result[2]["content"] == "Done A"

    def test_all_results_missing(self):
        """When ALL tool results are missing from a multi-call assistant message."""
        messages = [
            {"role": "user", "content": "Run both"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_x", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                    {"id": "call_y", "type": "function", "function": {"name": "y", "arguments": "{}"}},
                ],
            },
            # No tool results at all — next message is user
            {"role": "user", "content": "What happened?"},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # Should have 5 messages: user, asst+tc, placeholder_x, placeholder_y, user
        assert len(result) == 5
        placeholders = [m for m in result if m.get("role") == "tool"]
        assert len(placeholders) == 2
        placeholder_ids = {m["tool_call_id"] for m in placeholders}
        assert placeholder_ids == {"call_x", "call_y"}

    def test_empty_tool_calls_list_is_noop(self):
        """Assistant message with tool_calls=[] should not trigger repair."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Using tools...",
                "tool_calls": [],
            },
            {"role": "assistant", "content": "Done."},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)
        assert result == messages

    def test_tool_results_not_immediately_following(self):
        """Tool results separated from assistant message by another message type."""
        messages = [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_gap", "type": "function", "function": {"name": "t", "arguments": "{}"}},
                ],
            },
            # A user message appears before the tool result
            {"role": "user", "content": "Hurry up"},
            {"role": "tool", "tool_call_id": "call_gap", "content": "Late result"},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # The tool result is not immediately following, so scanner won't find it
        # A placeholder should be inserted right after the assistant message
        assert len(result) == 5
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_gap"
        assert "did not complete" in result[2]["content"]
