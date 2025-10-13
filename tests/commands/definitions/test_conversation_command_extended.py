"""Extended tests for the conversation command definition to improve coverage."""

import pytest
from unittest.mock import MagicMock, patch

from mcp_cli.commands.definitions.conversation import ConversationCommand
from mcp_cli.commands.base import CommandMode


@pytest.fixture
def conversation_command():
    """Create a conversation command instance."""
    return ConversationCommand()


@pytest.fixture
def mock_chat_context():
    """Create a mock chat context."""
    context = MagicMock()
    context.conversation_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    return context


def test_conversation_command_properties(conversation_command):
    """Test conversation command properties."""
    assert conversation_command.name == "conversation"
    assert conversation_command.aliases == ["history", "ch"]
    assert conversation_command.description == "Manage conversation history"
    assert conversation_command.modes == CommandMode.CHAT
    assert conversation_command.requires_context is True
    assert len(conversation_command.parameters) > 0
    assert "Manage conversation history" in conversation_command.help_text


@pytest.mark.asyncio
async def test_conversation_without_context(conversation_command):
    """Test conversation command without chat context."""
    result = await conversation_command.execute()

    assert result.success is False
    assert "Conversation command requires chat context" in result.error


@pytest.mark.asyncio
async def test_conversation_show_history(conversation_command, mock_chat_context):
    """Test showing conversation history."""
    mock_chat_context.conversation_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = await conversation_command.execute(chat_context=mock_chat_context)

    assert result.success is True
    assert "Conversation History" in result.output
    assert "Hello" in result.output
    assert "Hi there!" in result.output


@pytest.mark.asyncio
async def test_conversation_clear_history(conversation_command, mock_chat_context):
    """Test clearing conversation history."""
    mock_chat_context.clear_conversation = MagicMock()

    result = await conversation_command.execute(
        chat_context=mock_chat_context, action="clear"
    )

    assert result.success is True
    assert result.output == "Conversation history cleared."
    mock_chat_context.clear_conversation.assert_called_once()


@pytest.mark.asyncio
async def test_conversation_save_history(conversation_command, mock_chat_context):
    """Test saving conversation history."""
    mock_chat_context.conversation_history = [
        {"role": "user", "content": "Test message"}
    ]

    with patch("builtins.open", create=True) as mock_open:
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        result = await conversation_command.execute(
            chat_context=mock_chat_context, action="save", filename="conversation.json"
        )

        assert result.success is True
        assert "saved" in result.output.lower()
        mock_open.assert_called_once_with("conversation.json", "w")


@pytest.mark.asyncio
async def test_conversation_export_history(conversation_command, mock_chat_context):
    """Test exporting conversation history."""
    # Export is not a valid action in the actual implementation
    result = await conversation_command.execute(
        chat_context=mock_chat_context, action="export"
    )

    assert result.success is False
    assert "Unknown action" in result.error


@pytest.mark.asyncio
async def test_conversation_with_limit(conversation_command, mock_chat_context):
    """Test showing conversation with limit."""
    mock_chat_context.conversation_history = [
        {"role": "user", "content": f"Message {i}"} for i in range(10)
    ]

    # The actual implementation doesn't support limit parameter
    result = await conversation_command.execute(chat_context=mock_chat_context, limit=5)

    assert result.success is True
    # All messages will be shown as limit is not implemented
    assert "Conversation History" in result.output


@pytest.mark.asyncio
async def test_conversation_with_raw_format(conversation_command, mock_chat_context):
    """Test showing conversation in raw format."""
    mock_chat_context.conversation_history = [{"role": "user", "content": "Test"}]

    # Raw format is not supported in the actual implementation
    result = await conversation_command.execute(
        chat_context=mock_chat_context, format="raw"
    )

    assert result.success is True
    # Regular output is returned
    assert "Conversation History" in result.output


@pytest.mark.asyncio
async def test_conversation_invalid_action(conversation_command, mock_chat_context):
    """Test with invalid action."""
    result = await conversation_command.execute(
        chat_context=mock_chat_context, action="invalid"
    )

    assert result.success is False
    assert "Unknown action" in result.error


@pytest.mark.asyncio
async def test_conversation_save_error(conversation_command, mock_chat_context):
    """Test save with file error."""
    mock_chat_context.conversation_history = [{"role": "user", "content": "Test"}]

    with patch("builtins.open", side_effect=IOError("Cannot write file")):
        result = await conversation_command.execute(
            chat_context=mock_chat_context, action="save", filename="invalid/path.json"
        )

        assert result.success is False
        assert "Failed to save conversation" in result.error


@pytest.mark.asyncio
async def test_conversation_no_conversation_manager(conversation_command):
    """Test when context has no conversation_history attribute."""
    mock_context = MagicMock(spec=["some_attr"])

    result = await conversation_command.execute(chat_context=mock_context)

    assert result.success is False
    assert "Conversation history not available" in result.error


@pytest.mark.asyncio
async def test_conversation_empty_history(conversation_command, mock_chat_context):
    """Test with empty conversation history."""
    mock_chat_context.conversation_history = []

    result = await conversation_command.execute(chat_context=mock_chat_context)

    assert result.success is True
    assert result.output == "No conversation history."


@pytest.mark.asyncio
async def test_conversation_action_from_args(conversation_command, mock_chat_context):
    """Test getting action from args."""
    mock_chat_context.clear_conversation = MagicMock()

    result = await conversation_command.execute(
        chat_context=mock_chat_context, args=["clear"]
    )

    assert result.success is True
    assert result.output == "Conversation history cleared."
    mock_chat_context.clear_conversation.assert_called_once()
