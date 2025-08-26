"""Tests for interactive mode servers commands."""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from mcp_cli.interactive.commands.servers import (
    ServersCommand,
    ServersCapabilitiesCommand,
    ServersStatusCommand
)
from mcp_cli.tools.models import ServerInfo


class TestInteractiveServersCommand:
    """Test interactive mode servers command."""

    @pytest.fixture
    def servers_command(self):
        """Create ServersCommand instance."""
        return ServersCommand()

    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager."""
        tm = MagicMock()
        
        def make_server_info(id: int, name: str, tools: int, status: str):
            return ServerInfo(
                id=id,
                name=name,
                tool_count=tools,
                status=status,
                namespace="test"
            )
        
        tm.get_server_info = AsyncMock(return_value=[
            make_server_info(0, "sqlite", 6, "ready"),
            make_server_info(1, "filesystem", 10, "ready"),
        ])
        
        return tm

    def test_command_properties(self, servers_command):
        """Test ServersCommand basic properties."""
        assert servers_command.name == "servers"
        assert "srv" in servers_command.aliases
        assert "server" in servers_command.help.lower()

    def test_parse_arguments_basic(self, servers_command):
        """Test argument parsing for basic cases."""
        # No arguments
        options = servers_command._parse_arguments([])
        assert options["detailed"] is False
        assert options["capabilities"] is False
        assert options["transport"] is False
        assert options["format"] == "table"
        assert options["quiet"] is False

    def test_parse_arguments_detailed(self, servers_command):
        """Test parsing --detailed flag."""
        # Long form
        options = servers_command._parse_arguments(["--detailed"])
        assert options["detailed"] is True
        assert options["capabilities"] is True  # Auto-enabled
        assert options["transport"] is True     # Auto-enabled

        # Short form
        options = servers_command._parse_arguments(["-d"])
        assert options["detailed"] is True
        assert options["capabilities"] is True
        assert options["transport"] is True

    def test_parse_arguments_capabilities(self, servers_command):
        """Test parsing capability flags."""
        test_cases = [
            ["--capabilities"],
            ["--caps"],
            ["-c"],
        ]
        
        for args in test_cases:
            options = servers_command._parse_arguments(args)
            assert options["capabilities"] is True
            assert options["detailed"] is False  # Not auto-enabled
            assert options["transport"] is False

    def test_parse_arguments_transport(self, servers_command):
        """Test parsing transport flags."""
        test_cases = [
            ["--transport"],
            ["--trans"],
            ["-t"],
        ]
        
        for args in test_cases:
            options = servers_command._parse_arguments(args)
            assert options["transport"] is True
            assert options["detailed"] is False
            assert options["capabilities"] is False

    def test_parse_arguments_format(self, servers_command):
        """Test parsing format options."""
        # Long form with value
        options = servers_command._parse_arguments(["--format", "json"])
        assert options["format"] == "json"

        # Short form with value
        options = servers_command._parse_arguments(["-f", "tree"])
        assert options["format"] == "tree"

        # Direct format shortcut
        options = servers_command._parse_arguments(["json"])
        assert options["format"] == "json"

        # Invalid format
        options = servers_command._parse_arguments(["--format", "invalid"])
        assert options["invalid_format"] == "invalid"

    def test_parse_arguments_combined_short_flags(self, servers_command):
        """Test parsing combined short flags like -dct."""
        options = servers_command._parse_arguments(["-dct"])
        assert options["detailed"] is True
        assert options["capabilities"] is True
        assert options["transport"] is True

        options = servers_command._parse_arguments(["-ct"])
        assert options["detailed"] is False
        assert options["capabilities"] is True
        assert options["transport"] is True

    def test_parse_arguments_quiet(self, servers_command):
        """Test parsing quiet flag."""
        options = servers_command._parse_arguments(["--quiet"])
        assert options["quiet"] is True

        options = servers_command._parse_arguments(["-q"])
        assert options["quiet"] is True

    def test_parse_arguments_help(self, servers_command):
        """Test parsing help flag."""
        test_cases = [
            ["--help"],
            ["-h"],
            ["help"],
        ]
        
        for args in test_cases:
            options = servers_command._parse_arguments(args)
            assert options["help"] is True

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.servers_action_async")
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_basic(self, mock_output, mock_servers_action, servers_command, mock_tool_manager):
        """Test basic execution without arguments."""
        await servers_command.execute([], tool_manager=mock_tool_manager)
        
        mock_servers_action.assert_called_once_with(
            mock_tool_manager,
            detailed=False,
            show_capabilities=False,
            show_transport=False,
            output_format="table"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.servers_action_async")
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_detailed(self, mock_output, mock_servers_action, servers_command, mock_tool_manager):
        """Test execution with --detailed flag."""
        await servers_command.execute(["--detailed"], tool_manager=mock_tool_manager)
        
        mock_servers_action.assert_called_once_with(
            mock_tool_manager,
            detailed=True,
            show_capabilities=True,
            show_transport=True,
            output_format="table"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_no_tool_manager(self, mock_output, servers_command):
        """Test execution without tool manager."""
        await servers_command.execute([])
        
        # Should print error
        mock_output.print.assert_called_once()
        call_args = str(mock_output.print.call_args)
        assert "Error" in call_args
        assert "ToolManager not available" in call_args

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_invalid_format(self, mock_output, servers_command, mock_tool_manager):
        """Test execution with invalid format."""
        await servers_command.execute(["--format", "invalid"], tool_manager=mock_tool_manager)
        
        # Should print error about invalid format
        mock_output.print.assert_called_once()
        call_args = str(mock_output.print.call_args)
        assert "Error" in call_args
        assert "Invalid format" in call_args
        assert "invalid" in call_args

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.servers_action_async")
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_with_exception(self, mock_output, mock_servers_action, servers_command, mock_tool_manager):
        """Test execution handles exceptions gracefully."""
        mock_servers_action.side_effect = Exception("Connection error")
        
        await servers_command.execute([], tool_manager=mock_tool_manager)
        
        # Should print error message
        mock_output.print.assert_called()
        call_args = str(mock_output.print.call_args)
        assert "Error" in call_args
        assert "Failed to display server information" in call_args


class TestServersCapabilitiesCommand:
    """Test ServersCapabilitiesCommand."""

    @pytest.fixture
    def capabilities_command(self):
        """Create ServersCapabilitiesCommand instance."""
        return ServersCapabilitiesCommand()

    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager."""
        return MagicMock()

    def test_command_properties(self, capabilities_command):
        """Test command properties."""
        assert capabilities_command.name == "servers capabilities"
        assert "srv caps" in capabilities_command.aliases
        assert "capabilities" in capabilities_command.aliases
        assert "capability" in capabilities_command.help.lower()

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.servers_action_async")
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute(self, mock_output, mock_servers_action, capabilities_command, mock_tool_manager):
        """Test execution of capabilities command."""
        await capabilities_command.execute([], tool_manager=mock_tool_manager)
        
        # Should call with detailed capabilities and tree format
        mock_servers_action.assert_called_once_with(
            mock_tool_manager,
            detailed=True,
            show_capabilities=True,
            show_transport=False,
            output_format="tree"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_no_tool_manager(self, mock_output, capabilities_command):
        """Test execution without tool manager."""
        await capabilities_command.execute([])
        
        # Should print error
        mock_output.print.assert_called_once()
        call_args = str(mock_output.print.call_args)
        assert "Error" in call_args
        assert "ToolManager not available" in call_args


class TestServersStatusCommand:
    """Test ServersStatusCommand."""

    @pytest.fixture
    def status_command(self):
        """Create ServersStatusCommand instance."""
        return ServersStatusCommand()

    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager."""
        return MagicMock()

    def test_command_properties(self, status_command):
        """Test command properties."""
        assert status_command.name == "servers status"
        assert "srv status" in status_command.aliases
        assert "status" in status_command.aliases
        assert "status" in status_command.help.lower()

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.servers_action_async")
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute(self, mock_output, mock_servers_action, status_command, mock_tool_manager):
        """Test execution of status command."""
        await status_command.execute([], tool_manager=mock_tool_manager)
        
        # Should call with basic table format focused on status
        mock_servers_action.assert_called_once_with(
            mock_tool_manager,
            detailed=False,
            show_capabilities=False,
            show_transport=False,
            output_format="table"
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_no_tool_manager(self, mock_output, status_command):
        """Test execution without tool manager."""
        await status_command.execute([])
        
        # Should print error
        mock_output.print.assert_called_once()
        call_args = str(mock_output.print.call_args)
        assert "Error" in call_args
        assert "ToolManager not available" in call_args

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.servers_action_async")
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_with_exception(self, mock_output, mock_servers_action, status_command, mock_tool_manager):
        """Test execution handles exceptions gracefully."""
        mock_servers_action.side_effect = Exception("Network error")
        
        await status_command.execute([], tool_manager=mock_tool_manager)
        
        # Should print error message
        mock_output.print.assert_called()
        call_args = str(mock_output.print.call_args)
        assert "Error" in call_args
        assert "Failed to check server status" in call_args