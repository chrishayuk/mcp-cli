"""Tests for the ping command."""

import pytest
from unittest.mock import Mock, patch
from mcp_cli.commands.servers.ping import PingCommand


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

        # Test help_text
        assert "ping" in command.help_text.lower()
        assert "server" in command.help_text.lower()

        # Check parameters
        params = {p.name for p in command.parameters}
        assert "server_index" in params
        assert "all" in params
        assert "timeout" in params

    @pytest.mark.asyncio
    async def test_execute_all_servers(self, command):
        """Test pinging all servers."""
        from mcp_cli.tools.models import ServerInfo
        from unittest.mock import AsyncMock

        mock_tm = Mock()
        mock_server = ServerInfo(
            id=1,
            name="test-server",
            status="running",
            connected=True,
            tool_count=5,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server])

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_specific_server(self, command):
        """Test pinging a specific server."""
        from mcp_cli.tools.models import ServerInfo
        from unittest.mock import AsyncMock

        mock_tm = Mock()
        mock_server = ServerInfo(
            id=1,
            name="test-server",
            status="running",
            connected=True,
            tool_count=5,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server])

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm, server_index=0)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_no_tool_manager(self, command):
        """Test when no tool manager is available."""
        result = await command.execute()

        assert result.success is False
        assert "No active tool manager" in result.error

    @pytest.mark.asyncio
    async def test_execute_failed_ping(self, command):
        """Test when server is disconnected."""
        from mcp_cli.tools.models import ServerInfo
        from unittest.mock import AsyncMock

        mock_tm = Mock()
        mock_server = ServerInfo(
            id=1,
            name="test-server",
            status="stopped",
            connected=False,
            tool_count=0,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server])

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm)
            assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, command):
        """Test error handling during ping."""
        from unittest.mock import AsyncMock

        mock_tm = Mock()
        mock_tm.get_server_info = AsyncMock(side_effect=Exception("Network error"))

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm)
            assert result.success is False
            assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_context_exception(self, command):
        """Test getting tool manager from context when it throws exception."""
        with patch("mcp_cli.context.get_context") as mock_ctx:
            # Make context throw exception
            mock_ctx.side_effect = Exception("Context error")

            result = await command.execute()

            # Should handle exception and report no tool manager
            assert result.success is False
            assert "No active tool manager" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_args_list(self, command):
        """Test executing with args as list."""
        from mcp_cli.tools.models import ServerInfo
        from unittest.mock import AsyncMock

        mock_tm = Mock()
        mock_server1 = ServerInfo(
            id=1,
            name="server1",
            status="running",
            connected=True,
            tool_count=5,
            namespace="test",
        )
        mock_server2 = ServerInfo(
            id=2,
            name="server2",
            status="running",
            connected=True,
            tool_count=3,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server1, mock_server2])

        with patch("chuk_term.ui.output"):
            # Pass args as a list
            result = await command.execute(
                tool_manager=mock_tm, args=["server1", "server2"]
            )
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_args_string(self, command):
        """Test executing with args as string."""
        from mcp_cli.tools.models import ServerInfo
        from unittest.mock import AsyncMock

        mock_tm = Mock()
        mock_server = ServerInfo(
            id=1,
            name="server1",
            status="running",
            connected=True,
            tool_count=5,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server])

        with patch("chuk_term.ui.output"):
            # Pass args as a string
            result = await command.execute(tool_manager=mock_tm, args="server1")
            assert result.success is True
