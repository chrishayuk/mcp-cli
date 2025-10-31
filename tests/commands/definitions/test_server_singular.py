"""Tests for the server singular command."""

import pytest
from unittest.mock import patch
from mcp_cli.commands.definitions.server_singular import ServerSingularCommand


class TestServerSingularCommand:
    """Test the ServerSingularCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ServerSingularCommand instance."""
        return ServerSingularCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "server"
        assert command.aliases == []
        assert "Manage MCP servers" in command.description
        assert "server details" in command.help_text.lower()

    @pytest.mark.asyncio
    async def test_execute_no_args(self, command):
        """Test executing server command without args."""
        with patch(
            "mcp_cli.commands.actions.servers.servers_action_async"
        ) as mock_action:
            mock_action.return_value = []
            result = await command.execute(args=[])
            assert result.success is True
            mock_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_server_name(self, command):
        """Test executing server command with server name."""
        with patch(
            "mcp_cli.commands.actions.servers.servers_action_async"
        ) as mock_action:
            mock_action.return_value = []
            result = await command.execute(args=["test-server"])
            assert result.success is True
            # Should pass server name as args
            call_args = mock_action.call_args[0][0]
            assert call_args.args == ["test-server"]
