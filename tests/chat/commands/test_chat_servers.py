"""Tests for chat mode servers command."""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from mcp_cli.chat.commands.servers import servers_command
from mcp_cli.tools.models import ServerInfo


class TestChatServersCommand:
    """Test chat mode servers command."""

    @pytest.fixture
    def mock_context(self):
        """Create mock chat context with tool manager."""
        context = {}
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

        context["tool_manager"] = tool_manager
        context["config_path"] = "server_config.json"
        return context

    @pytest.fixture
    def mock_tool_manager_with_servers(self):
        """Create a tool manager with test servers."""
        tm = MagicMock()

        # Create test server info
        def make_server_info(id: int, name: str, tools: int, status: str):
            return ServerInfo(
                id=id, name=name, tool_count=tools, status=status, namespace="test"
            )

        tm.get_server_info = AsyncMock(
            return_value=[
                make_server_info(0, "sqlite", 6, "ready"),
                make_server_info(1, "filesystem", 10, "ready"),
                make_server_info(2, "github", 0, "error"),
            ]
        )

        # Mock get_adapted_tools_for_llm
        tm.get_adapted_tools_for_llm = AsyncMock(
            return_value=(
                [
                    {"function": {"name": f"tool_{i}", "description": f"Tool {i}"}}
                    for i in range(6)
                ],
                {},
            )
        )

        # Mock list_tools as fallback
        tm.list_tools = AsyncMock(
            return_value=[
                {"name": f"tool_{i}", "description": f"Tool {i}"} for i in range(6)
            ]
        )

        # Add servers attribute
        tm.servers = [
            MagicMock(name="sqlite"),
            MagicMock(name="filesystem"),
            MagicMock(name="github"),
        ]

        return tm

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_basic_servers_command(
        self, mock_pref_manager, mock_output, mock_context
    ):
        """Test basic /servers command without arguments."""
        # Setup preference manager
        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.return_value = False
        mock_pref_manager.return_value = pref_mgr

        # Execute command
        result = await servers_command(["/servers"], mock_context)

        # Should return True (command handled)
        assert result is True

        # Should print the servers list
        assert mock_output.rule.called
        assert mock_output.print.called

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    async def test_servers_no_tool_manager(self, mock_output):
        """Test /servers when tool manager is not available."""
        context = {"config_path": "server_config.json"}  # No tool_manager

        result = await servers_command(["/servers"], context)

        assert result is True

        # Should print error message
        mock_output.error.assert_called_once_with("ToolManager not available")

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_servers_help_command(
        self, mock_pref_manager, mock_output, mock_context
    ):
        """Test /servers help command."""
        result = await servers_command(["/servers", "help"], mock_context)

        assert result is True

        # Should show help
        mock_output.rule.assert_called_with("Servers Command Help")
        assert mock_output.print.called

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_servers_specific_server(
        self, mock_pref_manager, mock_output, mock_tool_manager_with_servers
    ):
        """Test /servers <name> to show specific server details."""
        # Setup preference manager
        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.return_value = False
        mock_pref_manager.return_value = pref_mgr

        context = {
            "tool_manager": mock_tool_manager_with_servers,
            "config_path": "server_config.json",
        }

        result = await servers_command(["/servers", "sqlite"], context)

        assert result is True

        # Should show either server details or error
        # Check if any output indicates server interaction
        assert mock_output.rule.called or mock_output.error.called

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.chat.commands.servers.toggle_server_status")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_servers_enable_command(
        self,
        mock_pref_manager,
        mock_toggle,
        mock_output,
        mock_tool_manager_with_servers,
    ):
        """Test /servers <name> enable command."""
        # Setup preference manager
        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.return_value = True  # Server is disabled
        mock_pref_manager.return_value = pref_mgr

        mock_toggle.return_value = True

        context = {
            "tool_manager": mock_tool_manager_with_servers,
            "config_path": "server_config.json",
        }

        result = await servers_command(["/servers", "sqlite", "enable"], context)

        assert result is True

        # Should call toggle_server_status
        mock_toggle.assert_called_once_with("sqlite", True)

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.chat.commands.servers.confirm")
    @patch("mcp_cli.chat.commands.servers.toggle_server_status")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_servers_disable_command(
        self,
        mock_pref_manager,
        mock_toggle,
        mock_confirm,
        mock_output,
        mock_tool_manager_with_servers,
    ):
        """Test /servers <name> disable command."""
        # Setup preference manager
        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.return_value = False  # Server is enabled
        mock_pref_manager.return_value = pref_mgr

        mock_confirm.return_value = True  # User confirms
        mock_toggle.return_value = True

        context = {
            "tool_manager": mock_tool_manager_with_servers,
            "config_path": "server_config.json",
        }

        result = await servers_command(["/servers", "sqlite", "disable"], context)

        assert result is True

        # Should ask for confirmation
        mock_confirm.assert_called_once_with("Disable server 'sqlite'?")

        # Should call toggle_server_status
        mock_toggle.assert_called_once_with("sqlite", False)

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_servers_config_command(
        self, mock_pref_manager, mock_output, mock_tool_manager_with_servers
    ):
        """Test /servers <name> config command."""
        # Setup preference manager
        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.return_value = False
        mock_pref_manager.return_value = pref_mgr

        context = {
            "tool_manager": mock_tool_manager_with_servers,
            "config_path": "server_config.json",
        }

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = """
            {
                "mcpServers": {
                    "sqlite": {
                        "command": "mcp-server-sqlite",
                        "args": ["--db-path", "test.db"]
                    }
                }
            }
            """

            result = await servers_command(["/servers", "sqlite", "config"], context)

        assert result is True

        # Should show configuration
        mock_output.rule.assert_called_with("Configuration: sqlite")

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_servers_tools_command(
        self, mock_pref_manager, mock_output, mock_tool_manager_with_servers
    ):
        """Test /servers <name> tools command."""
        # Setup preference manager
        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.return_value = False
        mock_pref_manager.return_value = pref_mgr

        context = {
            "tool_manager": mock_tool_manager_with_servers,
            "config_path": "server_config.json",
        }

        result = await servers_command(["/servers", "sqlite", "tools"], context)

        assert result is True

        # Should show tools
        mock_output.rule.assert_called_with("Available Tools: sqlite")

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_servers_ping_command(
        self, mock_pref_manager, mock_output, mock_tool_manager_with_servers
    ):
        """Test /servers <name> ping command."""
        # Setup preference manager
        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.return_value = False
        mock_pref_manager.return_value = pref_mgr

        context = {
            "tool_manager": mock_tool_manager_with_servers,
            "config_path": "server_config.json",
        }

        result = await servers_command(["/servers", "sqlite", "ping"], context)

        assert result is True

        # Should show ping result
        assert mock_output.info.called
        assert mock_output.success.called or mock_output.warning.called

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_servers_test_command(
        self, mock_pref_manager, mock_output, mock_tool_manager_with_servers
    ):
        """Test /servers <name> test command."""
        # Setup preference manager
        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.return_value = False
        mock_pref_manager.return_value = pref_mgr

        context = {
            "tool_manager": mock_tool_manager_with_servers,
            "config_path": "server_config.json",
        }

        result = await servers_command(["/servers", "sqlite", "test"], context)

        assert result is True

        # Should run tests
        mock_output.rule.assert_called_with("Testing 6 Tools")

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_servers_unknown_server(
        self, mock_pref_manager, mock_output, mock_tool_manager_with_servers
    ):
        """Test /servers with unknown server name."""
        # Setup preference manager
        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.return_value = False
        mock_pref_manager.return_value = pref_mgr

        context = {
            "tool_manager": mock_tool_manager_with_servers,
            "config_path": "server_config.json",
        }

        result = await servers_command(["/servers", "unknown_server"], context)

        assert result is True

        # Should show error
        mock_output.error.assert_called_with("Server 'unknown_server' not found")

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    async def test_servers_disabled_server_in_list(
        self, mock_pref_manager, mock_output, mock_tool_manager_with_servers
    ):
        """Test that disabled servers show correctly in list."""
        # Setup preference manager - sqlite is disabled
        pref_mgr = MagicMock()
        pref_mgr.is_server_disabled.side_effect = lambda name: name == "sqlite"
        mock_pref_manager.return_value = pref_mgr

        context = {
            "tool_manager": mock_tool_manager_with_servers,
            "config_path": "server_config.json",
        }

        result = await servers_command(["/servers"], context)

        assert result is True

        # Should show servers list with disabled status
        assert mock_output.print.called

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.servers.output")
    async def test_srv_alias(self, mock_output, mock_context):
        """Test /srv alias works (though registered separately)."""
        with patch(
            "mcp_cli.utils.preferences.get_preference_manager"
        ) as mock_pref_manager:
            pref_mgr = MagicMock()
            pref_mgr.is_server_disabled.return_value = False
            mock_pref_manager.return_value = pref_mgr

            result = await servers_command(["/srv"], mock_context)

            assert result is True
            assert mock_output.rule.called
