"""Tests for conversation load action to improve coverage."""

import pytest
from unittest.mock import MagicMock, patch

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
    """Test successful conversation load."""
    test_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    mock_chat_context.set_conversation_history = MagicMock()

    with patch("builtins.open"):
        with patch("json.load", return_value=test_history):
            result = await conversation_command.execute(
                chat_context=mock_chat_context,
                action="load",
                filename="conversation.json",
            )

            assert result.success is True
            assert "loaded" in result.output.lower()
            mock_chat_context.set_conversation_history.assert_called_once_with(
                test_history
            )


@pytest.mark.asyncio
async def test_conversation_load_from_args(conversation_command, mock_chat_context):
    """Test load with filename from args."""
    test_history = [{"role": "user", "content": "Test"}]
    mock_chat_context.set_conversation_history = MagicMock()

    with patch("builtins.open"):
        with patch("json.load", return_value=test_history):
            result = await conversation_command.execute(
                chat_context=mock_chat_context, args=["load", "test.json"]
            )

            assert result.success is True
            assert "loaded" in result.output.lower()


@pytest.mark.asyncio
async def test_conversation_load_no_filename(conversation_command, mock_chat_context):
    """Test load without filename."""
    result = await conversation_command.execute(
        chat_context=mock_chat_context, action="load"
    )

    assert result.success is False
    assert "Filename required for load" in result.error


@pytest.mark.asyncio
async def test_conversation_load_file_error(conversation_command, mock_chat_context):
    """Test load with file read error."""
    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        result = await conversation_command.execute(
            chat_context=mock_chat_context, action="load", filename="missing.json"
        )

        assert result.success is False
        assert "Failed to load conversation" in result.error


@pytest.mark.asyncio
async def test_conversation_load_no_set_method(conversation_command, mock_chat_context):
    """Test load when context has no set_conversation_history method."""
    test_history = [{"role": "user", "content": "Test"}]

    # Remove the set_conversation_history method
    del mock_chat_context.set_conversation_history

    with patch("builtins.open"):
        with patch("json.load", return_value=test_history):
            result = await conversation_command.execute(
                chat_context=mock_chat_context, action="load", filename="test.json"
            )

            assert result.success is False
            assert "Cannot set conversation history" in result.error


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
