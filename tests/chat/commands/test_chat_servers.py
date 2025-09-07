"""Tests for chat mode servers command."""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from mcp_cli.chat.commands.servers import (
    servers_command,
    display_servers_table,
)
from mcp_cli.tools.models import ServerInfo


class TestChatServersCommand:
    """Test chat mode servers command."""

    @pytest.fixture
    def mock_context(self):
        """Create mock chat context with tool manager."""
        context = MagicMock()
        tool_manager = MagicMock()

        # Mock get_server_info to return server data
        def make_server_info(id: int, name: str, tools: int, status: str):
            return ServerInfo(
                id=id, name=name, tool_count=tools, status=status, namespace="test"
            )

        tool_manager.get_server_info = AsyncMock(
            return_value=[
                make_server_info(0, "sqlite", 6, "ready"),
            ]
        )

        context.tool_manager = tool_manager
        return context

    @pytest.fixture
    def mock_tool_manager_with_servers(self):
        """Create a tool manager with test servers."""
        tm = MagicMock()

        # Create test servers
        servers = [
            ServerInfo(
                id=0, name="sqlite", tool_count=6, status="ready", namespace="test"
            ),
            ServerInfo(
                id=1, name="perplexity", tool_count=3, status="ready", namespace="test"
            ),
            ServerInfo(
                id=2, name="ios", tool_count=32, status="ready", namespace="test"
            ),
        ]

        # Mock the methods
        tm.get_server_info = AsyncMock(return_value=servers)
        tm.enable_server = AsyncMock(return_value=True)
        tm.disable_server = AsyncMock(return_value=True)

        # Mock server-specific operations
        tm.get_server_config = AsyncMock(
            return_value={"command": "mcp-server-sqlite", "args": ["test.db"]}
        )
        tm.get_server_tools = AsyncMock(
            return_value=[
                {"name": "list_tables", "description": "List all tables"},
                {"name": "query", "description": "Execute SQL query"},
            ]
        )
        tm.ping_server = AsyncMock(return_value={"latency": 5.2, "status": "ok"})
        tm.test_server = AsyncMock(
            return_value={
                "status": "ok",
                "tools": 6,
                "version": "1.0.0",
                "protocol": "mcp/1.0",
            }
        )

        # Mock servers attribute (list of servers)
        tm.servers = servers

        # Mock filtered list
        async def get_filtered_servers():
            return servers

        tm.get_filtered_servers = get_filtered_servers

        return tm

    @pytest.mark.asyncio
    @patch("mcp_cli.context.get_context")
    @patch("mcp_cli.chat.commands.servers.format_table")
    @patch("mcp_cli.chat.commands.servers.get_config")
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_basic_servers_command(
        self,
        mock_pref_manager,
        mock_output,
        mock_get_config,
        mock_format_table,
        mock_get_context,
        mock_tool_manager_with_servers,
    ):
        """Test basic /servers command."""
        # Setup
        context = MagicMock()
        context.tool_manager = mock_tool_manager_with_servers
        context.config_path = "server_config.json"
        mock_get_context.return_value = context
        mock_format_table.return_value = "formatted table"

        # Setup config
        mock_config = MagicMock()
        mock_config.servers = {}
        mock_get_config.return_value = mock_config

        # Setup preferences
        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.return_value = False
        mock_pref_manager.return_value = pref_mgr

        # Execute command
        result = await servers_command(["/servers"])

        assert result is True

        # Should create and print the servers table using format_table
        mock_format_table.assert_called_once()
        call_kwargs = mock_format_table.call_args.kwargs
        assert "data" in call_kwargs
        assert call_kwargs["title"] == "Available Servers"

        # Should print the formatted table
        mock_output.print.assert_any_call("formatted table")

        # Should check if tool manager is available
        context.tool_manager.get_server_info.assert_called_once()

    @pytest.mark.asyncio
    @patch("mcp_cli.context.get_context")
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.chat.commands.servers.get_config")
    async def test_servers_no_tool_manager(
        self, mock_get_config, mock_output, mock_get_context
    ):
        """Test /servers when tool manager is not available."""
        context = MagicMock()
        context.tool_manager = None
        context.config_path = "server_config.json"
        mock_get_context.return_value = context

        # Setup config
        mock_config = MagicMock()
        mock_config.servers = {}
        mock_get_config.return_value = mock_config

        result = await servers_command(["/servers"])

        assert result is True

        # Should print error message
        mock_output.error.assert_called_with("ToolManager not available")

    @pytest.mark.asyncio
    @patch("mcp_cli.context.get_context")
    @patch("mcp_cli.chat.commands.servers.format_table")
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.chat.commands.servers.get_config")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_servers_specific_server(
        self,
        mock_pref_manager,
        mock_get_config,
        mock_output,
        mock_format_table,
        mock_get_context,
        mock_tool_manager_with_servers,
    ):
        """Test /servers <name> command for specific server."""
        # Setup config
        mock_config = MagicMock()
        mock_config.servers = {}
        mock_get_config.return_value = mock_config
        mock_format_table.return_value = "formatted table"

        # Setup
        context = MagicMock()
        context.tool_manager = mock_tool_manager_with_servers
        mock_get_context.return_value = context

        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.return_value = False
        mock_pref_manager.return_value = pref_mgr

        # Execute command for specific server
        result = await servers_command(["/servers", "sqlite"])

        assert result is True

        # Should show server details using format_table
        mock_format_table.assert_called()
        mock_output.print.assert_called()

    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.chat.commands.servers.format_table")
    def test_display_servers_table(self, mock_format_table, mock_output):
        """Test displaying servers table."""
        mock_format_table.return_value = "formatted table"

        servers = [
            ServerInfo(
                id=0,
                name="sqlite",
                tool_count=6,
                status="ready",
                namespace="test",
            ),
            ServerInfo(
                id=1,
                name="filesystem",
                tool_count=4,
                status="ready",
                namespace="test",
            ),
        ]

        display_servers_table(servers, show_details=True)

        # Should create and print table
        mock_format_table.assert_called_once()
        call_kwargs = mock_format_table.call_args.kwargs
        assert "data" in call_kwargs
        assert len(call_kwargs["data"]) == 2

        mock_output.print.assert_any_call("formatted table")

    @pytest.mark.asyncio
    @patch("mcp_cli.context.get_context")
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.chat.commands.servers.get_config")
    async def test_servers_invalid_command(
        self, mock_get_config, mock_output, mock_get_context, mock_context
    ):
        """Test /servers with invalid subcommand."""
        mock_get_context.return_value = mock_context

        # Setup config
        mock_config = MagicMock()
        mock_config.servers = {}
        mock_get_config.return_value = mock_config

        # Execute invalid command
        result = await servers_command(["/servers", "invalid_command"])

        assert result is True

        # Should show error or warning
        assert (
            mock_output.error.called
            or mock_output.warning.called
            or mock_output.rule.called
        )
