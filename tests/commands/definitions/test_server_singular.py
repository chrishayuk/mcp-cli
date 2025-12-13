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
                id=1,
                name="test-server",
                status="running",
                connected=True,
                tool_count=5,
                namespace="test",
            )
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[mock_server]
            )
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["test-server"])
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_no_context(self, command):
        """Test executing when no context is available."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.return_value = None

            result = await command.execute(args=[])

            assert result.success is False
            assert "No tool manager available" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_tool_manager(self, command):
        """Test executing when context has no tool manager."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager = None

            result = await command.execute(args=[])

            assert result.success is False
            assert "No tool manager available" in result.error

    @pytest.mark.asyncio
    async def test_execute_args_as_string(self, command):
        """Test executing with args as string instead of list."""
        from mcp_cli.tools.models import ServerInfo

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_server = ServerInfo(
                id=1,
                name="test-server",
                status="running",
                connected=True,
                tool_count=5,
                namespace="test",
            )
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[mock_server]
            )
            with patch("chuk_term.ui.output"):
                result = await command.execute(args="test-server")
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_servers(self, command):
        """Test listing servers when servers exist."""
        from mcp_cli.tools.models import ServerInfo

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_server = ServerInfo(
                id=1,
                name="test-server",
                status="running",
                connected=True,
                tool_count=5,
                namespace="test",
            )
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[mock_server]
            )
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table") as mock_format:
                    mock_format.return_value = "table"
                    result = await command.execute(args=[])
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_command(self, command):
        """Test 'list' subcommand."""
        from mcp_cli.tools.models import ServerInfo

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_server = ServerInfo(
                id=1,
                name="test-server",
                status="running",
                connected=True,
                tool_count=5,
                namespace="test",
            )
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[mock_server]
            )
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table") as mock_format:
                    mock_format.return_value = "table"
                    result = await command.execute(args=["list"])
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_no_servers(self, command):
        """Test 'list' subcommand with no servers."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[])
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["list"])
                assert result.success is True
                assert "No servers connected" in result.output

    @pytest.mark.asyncio
    async def test_execute_add_command(self, command):
        """Test 'add' subcommand (not implemented)."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[])
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["add", "myserver"])
                assert result.success is False
                assert "not yet implemented" in result.error

    @pytest.mark.asyncio
    async def test_execute_remove_command(self, command):
        """Test 'remove' subcommand (not implemented)."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[])
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["remove", "myserver"])
                assert result.success is False
                assert "not yet implemented" in result.error

    @pytest.mark.asyncio
    async def test_execute_enable_command(self, command):
        """Test 'enable' subcommand (not implemented)."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[])
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["enable", "myserver"])
                assert result.success is False
                assert "not yet implemented" in result.error

    @pytest.mark.asyncio
    async def test_execute_disable_command(self, command):
        """Test 'disable' subcommand (not implemented)."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[])
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["disable", "myserver"])
                assert result.success is False
                assert "not yet implemented" in result.error

    @pytest.mark.asyncio
    async def test_execute_ping_command(self, command):
        """Test 'ping' subcommand (not implemented)."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[])
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["ping", "myserver"])
                assert result.success is False
                assert "not yet implemented" in result.error

    @pytest.mark.asyncio
    async def test_execute_server_not_found(self, command):
        """Test when specified server is not found."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[])
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["nonexistent-server"])
                assert result.success is False
                assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_list_exception(self, command):
        """Test list command when exception occurs."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                side_effect=Exception("Database error")
            )
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=[])
                assert result.success is False
                assert "Failed to list servers" in result.error

    @pytest.mark.asyncio
    async def test_execute_list_subcommand_exception(self, command):
        """Test 'list' subcommand when exception occurs."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                side_effect=Exception("Database error")
            )
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["list"])
                assert result.success is False
                assert "Failed to list servers" in result.error

    @pytest.mark.asyncio
    async def test_execute_server_details_exception(self, command):
        """Test server details when exception occurs."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                side_effect=Exception("API error")
            )
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["some-server"])
                assert result.success is False
                assert "Failed to get server details" in result.error

    @pytest.mark.asyncio
    async def test_execute_disconnected_server(self, command):
        """Test listing with a disconnected server."""
        from mcp_cli.tools.models import ServerInfo

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_server = ServerInfo(
                id=1,
                name="test-server",
                status="stopped",
                connected=False,
                tool_count=0,
                namespace="test",
            )
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[mock_server]
            )
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table") as mock_format:
                    mock_format.return_value = "table"
                    result = await command.execute(args=[])
                    assert result.success is True
