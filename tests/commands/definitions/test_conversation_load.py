"""Tests for conversation load action to improve coverage."""

import pytest
from unittest.mock import MagicMock

from mcp_cli.commands.conversation.conversation import ConversationCommand
from mcp_cli.chat.models import Message, MessageRole


@pytest.fixture
def conversation_command():
    """Create a conversation command instance."""
    return ConversationCommand()


@pytest.fixture
def mock_chat_context():
    """Create a mock chat context."""
    context = MagicMock()
    context.conversation_history = []
    return context


@pytest.mark.asyncio
async def test_conversation_load_success(conversation_command, mock_chat_context):
    """Test successful conversation load via session persistence."""
    mock_chat_context.load_session = MagicMock(return_value=True)

    result = await conversation_command.execute(
        chat_context=mock_chat_context,
        action="load",
        filename="session_abc123",
    )

    assert result.success is True
    assert "loaded" in result.output.lower()
    mock_chat_context.load_session.assert_called_once_with("session_abc123")


@pytest.mark.asyncio
async def test_conversation_load_from_args(conversation_command, mock_chat_context):
    """Test load with session_id from args."""
    mock_chat_context.load_session = MagicMock(return_value=True)

    result = await conversation_command.execute(
        chat_context=mock_chat_context, args=["load", "session_xyz"]
    )

    assert result.success is True
    assert "loaded" in result.output.lower()
    mock_chat_context.load_session.assert_called_once_with("session_xyz")


@pytest.mark.asyncio
async def test_conversation_load_no_filename(conversation_command, mock_chat_context):
    """Test load without session ID."""
    result = await conversation_command.execute(
        chat_context=mock_chat_context, action="load"
    )

    assert result.success is False
    assert "Session ID required" in result.error


@pytest.mark.asyncio
async def test_conversation_load_failure(conversation_command, mock_chat_context):
    """Test load when load_session returns False."""
    mock_chat_context.load_session = MagicMock(return_value=False)

    result = await conversation_command.execute(
        chat_context=mock_chat_context, action="load", filename="bad_session"
    )

    assert result.success is False
    assert "Failed to load session" in result.error


@pytest.mark.asyncio
async def test_conversation_load_no_session_support(
    conversation_command, mock_chat_context
):
    """Test load when context has no load_session method."""
    # Remove load_session so hasattr returns False
    mock_chat_context.configure_mock(**{"load_session": None})
    del mock_chat_context.load_session

    result = await conversation_command.execute(
        chat_context=mock_chat_context, action="load", filename="session_123"
    )

    assert result.success is False
    assert "not available" in result.error.lower()


@pytest.mark.asyncio
async def test_conversation_action_from_string_arg(
    conversation_command, mock_chat_context
):
    """Test action extraction from string arg."""
    mock_chat_context.clear_conversation = MagicMock()

    result = await conversation_command.execute(
        chat_context=mock_chat_context,
        args="clear",  # Single string instead of list
    )

    assert result.success is True
    assert result.output == "Conversation history cleared."
    mock_chat_context.clear_conversation.assert_called_once()


@pytest.mark.asyncio
async def test_conversation_truncate_long_message(
    conversation_command, mock_chat_context
):
    """Test that long messages get truncated in show."""
    long_message = "A" * 250  # Create a message longer than 200 chars
    mock_chat_context.conversation_history = [
        Message(role=MessageRole.USER, content=long_message)
    ]

    result = await conversation_command.execute(
        chat_context=mock_chat_context, action="show"
    )

    assert result.success is True
    assert "..." in result.output  # Check that truncation happened
    assert len(result.output.split("\n")[3]) < 250  # Truncated line should be shorter
