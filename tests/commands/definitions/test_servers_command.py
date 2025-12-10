"""Tests for the servers command."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from mcp_cli.commands.servers.servers import ServersCommand


class ServerInfo:
    """Mock ServerInfo class."""
    def __init__(self, name, status="connected", tool_count=0):
        self.name = name
        self.status = status
        self.tool_count = tool_count
        self.transport = MagicMock()
        self.transport.value = "stdio"
        self.connected = True
        self.namespace = "default"


class TestServersCommand:
    """Test the ServersCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ServersCommand instance."""
        return ServersCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "servers"
        assert command.aliases == []
        assert command.description == "List connected MCP servers and their status"
        assert "List connected MCP servers" in command.help_text

        # Check parameters
        params = {p.name for p in command.parameters}
        assert "detailed" in params
        assert "format" in params
        assert "ping" in params

    @pytest.mark.asyncio
    async def test_execute_basic(self, command):
        """Test basic execution without parameters."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[
                ServerInfo("test-server", "connected", 5)
            ])

            with patch("chuk_term.ui.output"):
                result = await command.execute()

            # Check result
            assert result.success is True
            mock_ctx.tool_manager.get_server_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_detailed(self, command):
        """Test execution with detailed flag."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[
                ServerInfo("test-server", "connected", 5)
            ])

            with patch("chuk_term.ui.output"):
                result = await command.execute(detailed=True)

            assert result.success is True
            mock_ctx.tool_manager.get_server_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_format(self, command):
        """Test execution with different output formats."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[])

            with patch("chuk_term.ui.output"):
                # Test with json format
                result = await command.execute(format="json")

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, command):
        """Test error handling during execution."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(side_effect=Exception("Connection failed"))

            result = await command.execute()

            assert result.success is False
            assert "Connection failed" in result.error or "Failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_servers(self, command):
        """Test execution when no servers are connected."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[])

            result = await command.execute()

            assert result.success is True
            # The result should indicate no servers are connected

    def test_parameter_validation(self, command):
        """Test parameter validation."""
        # Test with valid format
        error = command.validate_parameters(format="table")
        assert error is None

        error = command.validate_parameters(format="json")
        assert error is None

        # Test with invalid format
        error = command.validate_parameters(format="invalid")
        assert error is not None
        assert "Invalid choice" in error
