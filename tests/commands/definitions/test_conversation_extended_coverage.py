"""Extended coverage tests for conversation command."""

import pytest
from unittest.mock import MagicMock, patch

from mcp_cli.commands.definitions.conversation import ConversationCommand
from mcp_cli.chat.models import Message, MessageRole


@pytest.fixture
def conversation_command():
    """Create a conversation command instance."""
    return ConversationCommand()


@pytest.fixture
def mock_chat_context():
    """Create a mock chat context with tool calls."""
    context = MagicMock()
    context.conversation_history = [
        Message(role=MessageRole.USER, content="Hello"),
        Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        Message(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                {"function": {"name": "test_tool", "arguments": '{"param": "value"}'}}
            ],
        ),
        Message(role=MessageRole.TOOL, content="Tool result"),
        Message(role=MessageRole.SYSTEM, content="System message"),
    ]
    context.clear_conversation = MagicMock()
    return context


@pytest.mark.asyncio
async def test_conversation_detail_view_with_tool_calls(
    conversation_command, mock_chat_context
):
    """Test detailed view of a message with tool calls."""
    # View the tool call message (row 3)
    result = await conversation_command.execute(
        chat_context=mock_chat_context, args=["3"]
    )

    assert result.success is True
    # The panel should be displayed with tool call information


@pytest.mark.asyncio
async def test_conversation_detail_view_invalid_row(
    conversation_command, mock_chat_context
):
    """Test detailed view with invalid row number."""
    result = await conversation_command.execute(
        chat_context=mock_chat_context,
        args=["10"],  # Out of range
    )

    assert result.success is False
    assert "Invalid row number" in result.error
    assert "Valid range: 1-5" in result.error


@pytest.mark.asyncio
async def test_conversation_detail_view_with_no_content(conversation_command):
    """Test detailed view when message has no content and no tool calls."""
    context = MagicMock()
    context.conversation_history = [
        Message(role=MessageRole.ASSISTANT, content=None)  # No content, no tool_calls
    ]

    result = await conversation_command.execute(chat_context=context, args=["1"])

    assert result.success is True
    # Should display "[No content]"


@pytest.mark.asyncio
async def test_conversation_table_with_all_role_types(conversation_command):
    """Test table display with all role types."""
    context = MagicMock()
    context.conversation_history = [
        Message(role=MessageRole.SYSTEM, content="System message"),
        Message(role=MessageRole.USER, content="User message"),
        Message(role=MessageRole.ASSISTANT, content="Assistant message"),
        Message(role=MessageRole.TOOL, content="Tool result"),
    ]

    result = await conversation_command.execute(chat_context=context)

    assert result.success is True
    assert result.data is not None
    assert len(result.data) == 4

    # Check role displays
    roles = [item["Role"] for item in result.data]
    assert "🔧 System" in roles
    assert "👤 User" in roles
    assert "🤖 Assistant" in roles
    assert "🔨 Tool" in roles


@pytest.mark.asyncio
async def test_conversation_save_with_args_filename(
    conversation_command, mock_chat_context
):
    """Test save with filename from args."""
    with patch("builtins.open", create=True) as mock_open:
        with patch("json.dump"):
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            result = await conversation_command.execute(
                chat_context=mock_chat_context,
                action="save",
                args=["save", "test_file.json"],
            )

            assert result.success is True
            assert "saved to test_file.json" in result.output.lower()
            mock_open.assert_called_once_with("test_file.json", "w")


@pytest.mark.asyncio
async def test_conversation_save_no_filename(conversation_command, mock_chat_context):
    """Test save without filename."""
    result = await conversation_command.execute(
        chat_context=mock_chat_context, action="save"
    )

    assert result.success is False
    assert "Filename required" in result.error


@pytest.mark.asyncio
async def test_conversation_clear_no_method(conversation_command):
    """Test clear when context has no clear_conversation method."""
    context = MagicMock(spec=["conversation_history"])
    context.conversation_history = [Message(role=MessageRole.USER, content="test")]

    result = await conversation_command.execute(chat_context=context, action="clear")

    assert result.success is False
    assert "Cannot clear conversation" in result.error


@pytest.mark.asyncio
async def test_conversation_save_no_history_attribute(conversation_command):
    """Test save when context has no conversation_history."""
    context = MagicMock(spec=[])

    result = await conversation_command.execute(
        chat_context=context, action="save", filename="test.json"
    )

    assert result.success is False
    assert "Conversation history not available" in result.error


@pytest.mark.asyncio
async def test_conversation_with_tool_calls_in_table(conversation_command):
    """Test table display with tool call messages."""
    context = MagicMock()
    context.conversation_history = [
        Message(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[{"function": {"name": "tool1"}}],
        )
    ]

    result = await conversation_command.execute(chat_context=context)

    assert result.success is True
    assert result.data is not None
    # In test mode, should see tool call placeholder
    assert "[Tool call - see /toolhistory]" in result.output or "Tool call" in str(
        result.data
    )


@pytest.mark.asyncio
async def test_conversation_row_parsing_from_string_arg(
    conversation_command, mock_chat_context
):
    """Test parsing row number from string argument."""
    result = await conversation_command.execute(
        chat_context=mock_chat_context,
        args="2",  # String instead of list
    )

    assert result.success is True
    # Should show detail view for row 2


@pytest.mark.asyncio
async def test_conversation_action_from_string_arg(
    conversation_command, mock_chat_context
):
    """Test parsing action from string argument."""
    mock_chat_context.clear_conversation = MagicMock()

    result = await conversation_command.execute(
        chat_context=mock_chat_context,
        args="clear",  # String action
    )

    assert result.success is True
    assert result.output == "Conversation history cleared."
    mock_chat_context.clear_conversation.assert_called_once()


@pytest.mark.asyncio
async def test_conversation_unknown_action(conversation_command, mock_chat_context):
    """Test with completely unknown action."""
    result = await conversation_command.execute(
        chat_context=mock_chat_context, action="unknown_action"
    )

    assert result.success is False
    assert "Unknown action: unknown_action" in result.error


@pytest.mark.asyncio
async def test_conversation_export_as_action(conversation_command, mock_chat_context):
    """Test export action (should be unknown)."""
    result = await conversation_command.execute(
        chat_context=mock_chat_context, action="export"
    )

    assert result.success is False
    assert "Unknown action: export" in result.error
