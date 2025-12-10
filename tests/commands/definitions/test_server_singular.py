"""Tests for the server singular command."""

import pytest
from unittest.mock import patch, AsyncMock
from mcp_cli.commands.servers.server_singular import ServerSingularCommand


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
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[])
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await command.execute(args=[])
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_server_name(self, command):
        """Test executing server command with server name."""
        from mcp_cli.tools.models import ServerInfo
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_server = ServerInfo(
                id=1, name="test-server", status="running",
                connected=True, tool_count=5, namespace="test"
            )
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[mock_server])
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["test-server"])
                assert result.success is True
