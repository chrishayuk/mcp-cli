# tests/chat/commands/test_chat_resources.py
"""Tests for the chat /resources command."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from mcp_cli.chat.commands.resources import cmd_resources
from tests.conftest import setup_test_context


class TestChatResourcesCommand:
    """Test the /resources command in chat mode."""

    @pytest.fixture(autouse=True)
    def setup_context(self):
        """Set up context before each test."""
        # Create a mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.list_resources = AsyncMock(return_value=[])

        context = setup_test_context(tool_manager=mock_tool_manager)
        yield context

    @pytest.mark.asyncio
    async def test_resources_command_no_tool_manager(self, setup_context):
        """Test /resources when no tool manager is available."""
        # Remove tool manager
        setup_context.tool_manager = None

        with patch("mcp_cli.chat.commands.resources.output") as mock_output:
            result = await cmd_resources([])

            # Should show error and return True (command handled)
            assert result is True
            mock_output.error.assert_called_with("ToolManager not available.")

    @pytest.mark.asyncio
    async def test_resources_command_delegates_to_action(self, setup_context):
        """Test that /resources delegates to resources_action_async."""
        with patch(
            "mcp_cli.chat.commands.resources.resources_action_async"
        ) as mock_action:
            mock_action.return_value = []

            result = await cmd_resources([])

            # Should call the action and return True
            assert result is True
            mock_action.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_resources_command_with_resources(self, setup_context):
        """Test /resources when resources are available."""
        # Set up mock resources
        mock_resources = [
            {
                "server": "sqlite",
                "uri": "/tmp/data.db",
                "size": 1024,
                "mimeType": "application/x-sqlite3",
            },
            {
                "server": "files",
                "uri": "/tmp/report.csv",
                "size": 500,
                "mimeType": "text/csv",
            },
        ]
        setup_context.tool_manager.list_resources.return_value = mock_resources

        with patch(
            "mcp_cli.chat.commands.resources.resources_action_async"
        ) as mock_action:
            mock_action.return_value = mock_resources

            result = await cmd_resources([])

            # Should succeed
            assert result is True
            mock_action.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_resources_command_handles_args(self, setup_context):
        """Test that /resources ignores any arguments passed."""
        with patch(
            "mcp_cli.chat.commands.resources.resources_action_async"
        ) as mock_action:
            mock_action.return_value = []

            # Pass some random arguments - they should be ignored
            result = await cmd_resources(["some", "random", "args"])

            # Should still work normally
            assert result is True
            mock_action.assert_called_once_with()
