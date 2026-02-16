"""Tests for the conversation command."""

import pytest
from unittest.mock import Mock
from mcp_cli.commands.conversation.conversation import ConversationCommand
from mcp_cli.commands.base import CommandMode
from mcp_cli.chat.models import Message, MessageRole


class TestConversationCommand:
    """Test the ConversationCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ConversationCommand instance."""
        return ConversationCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "conversation"
        assert command.aliases == ["history", "ch"]
        assert "conversation" in command.description.lower()
        assert command.modes == CommandMode.CHAT  # Chat mode only

        # Check parameters
        params = {p.name for p in command.parameters}
        assert "action" in params
        assert "filename" in params

    @pytest.mark.asyncio
    async def test_execute_show(self, command):
        """Test showing conversation history."""
        mock_context = Mock()
        mock_context.conversation_history = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
            Message(role=MessageRole.USER, content="How are you?"),
            Message(role=MessageRole.ASSISTANT, content="I'm doing well, thank you!"),
        ]

        result = await command.execute(chat_context=mock_context, action="show")

        assert result.success is True
        assert result.output is not None
        # Output should contain conversation messages
        assert "Hello" in result.output
        assert "Hi there!" in result.output

    @pytest.mark.asyncio
    async def test_execute_clear(self, command):
        """Test clearing conversation history."""
        mock_context = Mock()
        mock_context.conversation_history = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        ]

        result = await command.execute(chat_context=mock_context, action="clear")

        assert result.success is True
        # Conversation should be cleared
        mock_context.clear_conversation.assert_called_once()
        assert "cleared" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_save(self, command):
        """Test saving conversation via session persistence."""
        mock_context = Mock()
        mock_context.conversation_history = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        ]
        mock_context.save_session = Mock(return_value="/tmp/session_abc123")

        result = await command.execute(
            chat_context=mock_context,
            action="save",
        )

        assert result.success is True
        mock_context.save_session.assert_called_once()
        assert "saved" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_load(self, command):
        """Test loading conversation via session persistence."""
        mock_context = Mock()
        mock_context.load_session = Mock(return_value=True)

        result = await command.execute(
            chat_context=mock_context,
            action="load",
            filename="session_abc123",
        )

        assert result.success is True
        mock_context.load_session.assert_called_once_with("session_abc123")
        assert "loaded" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_empty_history(self, command):
        """Test showing empty conversation history."""
        mock_context = Mock()
        mock_context.conversation_history = []

        result = await command.execute(chat_context=mock_context, action="show")

        assert result.success is True
        assert "No conversation history" in result.output

    @pytest.mark.asyncio
    async def test_execute_invalid_action(self, command):
        """Test with invalid action."""
        mock_context = Mock()

        result = await command.execute(chat_context=mock_context, action="invalid")

        assert result.success is False
        assert "Unknown action" in result.error or "Invalid" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_context(self, command):
        """Test execution without context."""
        result = await command.execute(action="show")

        assert result.success is False
        assert "context" in result.error.lower()
