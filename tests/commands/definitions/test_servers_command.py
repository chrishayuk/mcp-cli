"""Tests for the servers command."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from mcp_cli.commands.servers.servers import ServersCommand
from mcp_cli.tools.models import ServerInfo, TransportType


def create_server_info(
    name: str,
    status: str = "connected",
    tool_count: int = 0,
    transport: TransportType = TransportType.STDIO,
    connected: bool = True,
) -> ServerInfo:
    """Create a ServerInfo instance for testing."""
    return ServerInfo(
        id=0,
        name=name,
        status=status,
        tool_count=tool_count,
        namespace=name,
        enabled=True,
        connected=connected,
        transport=transport,
        capabilities={},
    )


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
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[create_server_info("test-server", "connected", 5)]
            )

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
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[create_server_info("test-server", "connected", 5)]
            )

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
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                side_effect=Exception("Connection failed")
            )

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

    @pytest.mark.asyncio
    async def test_execute_no_context(self, command):
        """Test execution when no context is available."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.return_value = None

            result = await command.execute()

            assert result.success is False
            assert "No tool manager available" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_tool_manager(self, command):
        """Test execution when context has no tool manager."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = MagicMock()
            mock_ctx.tool_manager = None
            mock_get_ctx.return_value = mock_ctx

            result = await command.execute()

            assert result.success is False
            assert "No tool manager available" in result.error

    @pytest.mark.asyncio
    async def test_execute_json_format_with_servers(self, command):
        """Test JSON format output with servers."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_server = create_server_info("test-server", "connected", 5)
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[mock_server]
            )

            with patch("chuk_term.ui.output"):
                result = await command.execute(format="json")

            assert result.success is True
            assert result.data is not None
            assert len(result.data) == 1
            assert result.data[0]["name"] == "test-server"
            assert result.data[0]["tool_count"] == 5

    @pytest.mark.asyncio
    async def test_execute_with_ping_flag(self, command):
        """Test execution with ping flag."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_server = create_server_info("test-server", "connected", 5)
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[mock_server]
            )

            with patch("chuk_term.ui.output") as mock_output:
                result = await command.execute(ping=True)

            assert result.success is True
            # Verify ping info was called
            mock_output.info.assert_called()

    @pytest.mark.asyncio
    async def test_execute_disconnected_server(self, command):
        """Test execution with a disconnected server."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_server = create_server_info(
                "test-server", "stopped", 0, connected=False
            )
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[mock_server]
            )

            with patch("chuk_term.ui.output"):
                result = await command.execute()

            assert result.success is True
            # Table output doesn't populate result.data, but command succeeds

    @pytest.mark.asyncio
    async def test_execute_server_with_unknown_transport(self, command):
        """Test execution with a server that has unknown transport."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_server = create_server_info(
                "test-server", "connected", 5, transport=TransportType.UNKNOWN
            )
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[mock_server]
            )

            with patch("chuk_term.ui.output"):
                # Test detailed view to exercise transport/namespace columns
                result = await command.execute(detailed=True)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_json_server_with_unknown_transport(self, command):
        """Test JSON output with a server that has unknown transport."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_server = create_server_info(
                "test-server", "connected", 5, transport=TransportType.UNKNOWN
            )
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[mock_server]
            )

            with patch("chuk_term.ui.output"):
                result = await command.execute(format="json")

            assert result.success is True
            assert result.data[0]["transport"] == "unknown"


class TestServersConnectionInfo:
    """Test _get_connection_info method."""

    @pytest.fixture
    def command(self):
        """Create a ServersCommand instance."""
        return ServersCommand()

    def test_connection_info_stdio_with_command(self, command):
        """Test connection info for STDIO with command."""
        server = create_server_info("test", transport=TransportType.STDIO)
        server.command = "python"
        server.args = None

        result = command._get_connection_info(server)

        assert result == "python"

    def test_connection_info_stdio_with_args(self, command):
        """Test connection info for STDIO with command and args."""
        server = create_server_info("test", transport=TransportType.STDIO)
        server.command = "python"
        server.args = ["-m", "server"]

        result = command._get_connection_info(server)

        assert result == "python -m server"

    def test_connection_info_stdio_with_many_args(self, command):
        """Test connection info for STDIO with command and many args (truncated)."""
        server = create_server_info("test", transport=TransportType.STDIO)
        server.command = "python"
        server.args = ["-m", "server", "--verbose", "--debug"]

        result = command._get_connection_info(server)

        assert result == "python -m server ..."

    def test_connection_info_stdio_no_command(self, command):
        """Test connection info for STDIO without command."""
        server = create_server_info("test", transport=TransportType.STDIO)
        server.command = None

        result = command._get_connection_info(server)

        assert result == "stdio"

    def test_connection_info_http_with_url(self, command):
        """Test connection info for HTTP with URL."""
        server = create_server_info("test", transport=TransportType.HTTP)
        server.url = "http://localhost:8080"

        result = command._get_connection_info(server)

        assert result == "http://localhost:8080"

    def test_connection_info_http_with_long_url(self, command):
        """Test connection info for HTTP with long URL (truncated)."""
        server = create_server_info("test", transport=TransportType.HTTP)
        server.url = "http://localhost:8080/very/long/path/to/api/endpoint"

        result = command._get_connection_info(server)

        assert result.endswith("...")
        assert len(result) == 40

    def test_connection_info_http_no_url(self, command):
        """Test connection info for HTTP without URL."""
        server = create_server_info("test", transport=TransportType.HTTP)
        server.url = None

        result = command._get_connection_info(server)

        assert result == "http"

    def test_connection_info_sse_with_url(self, command):
        """Test connection info for SSE with URL."""
        server = create_server_info("test", transport=TransportType.SSE)
        server.url = "http://localhost:8080/sse"

        result = command._get_connection_info(server)

        assert result == "http://localhost:8080/sse"

    def test_connection_info_sse_no_url(self, command):
        """Test connection info for SSE without URL."""
        server = create_server_info("test", transport=TransportType.SSE)
        server.url = None

        result = command._get_connection_info(server)

        assert result == "sse"

    def test_connection_info_unknown_transport(self, command):
        """Test connection info for unknown transport."""
        server = create_server_info("test", transport=TransportType.UNKNOWN)

        result = command._get_connection_info(server)

        assert result == "unknown"
