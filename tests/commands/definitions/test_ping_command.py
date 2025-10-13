"""Tests for the ping command."""

import pytest
from unittest.mock import Mock, patch
from mcp_cli.commands.definitions.ping import PingCommand


class TestPingCommand:
    """Test the PingCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a PingCommand instance."""
        return PingCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "ping"
        assert command.aliases == []
        assert "Test connectivity to MCP servers" in command.description

        # Check parameters
        params = {p.name for p in command.parameters}
        assert "server_index" in params
        assert "all" in params
        assert "timeout" in params

    @pytest.mark.asyncio
    async def test_execute_all_servers(self, command):
        """Test pinging all servers."""
        mock_tm = Mock()

        with patch("mcp_cli.commands.actions.ping.ping_action_async") as mock_ping:
            mock_ping.return_value = True  # Success

            result = await command.execute(tool_manager=mock_tm)

            # Verify ping was called without targets (ping all)
            mock_ping.assert_called_once_with(mock_tm, targets=[])

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_specific_server(self, command):
        """Test pinging a specific server."""
        mock_tm = Mock()

        with patch("mcp_cli.commands.actions.ping.ping_action_async") as mock_ping:
            mock_ping.return_value = True

            result = await command.execute(tool_manager=mock_tm, server_index=1)

            # Verify ping was called with server target
            mock_ping.assert_called_once_with(mock_tm, targets=["1"])

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_no_tool_manager(self, command):
        """Test when no tool manager is available."""
        result = await command.execute()

        assert result.success is False
        assert "No active tool manager" in result.error

    @pytest.mark.asyncio
    async def test_execute_failed_ping(self, command):
        """Test when ping returns False."""
        mock_tm = Mock()

        with patch("mcp_cli.commands.actions.ping.ping_action_async") as mock_ping:
            mock_ping.return_value = False  # No servers pinged

            result = await command.execute(tool_manager=mock_tm)

            assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, command):
        """Test error handling during ping."""
        mock_tm = Mock()

        with patch("mcp_cli.commands.actions.ping.ping_action_async") as mock_ping:
            mock_ping.side_effect = Exception("Network error")

            result = await command.execute(tool_manager=mock_tm)

            assert result.success is False
            assert "Network error" in result.error
