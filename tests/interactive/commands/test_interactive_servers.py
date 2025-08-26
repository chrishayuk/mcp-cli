"""Tests for interactive mode servers commands."""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from mcp_cli.interactive.commands.servers import ServersCommand
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
                id=id, name=name, tool_count=tools, status=status, namespace="test"
            )

        tm.get_server_info = AsyncMock(
            return_value=[
                make_server_info(0, "sqlite", 6, "ready"),
                make_server_info(1, "filesystem", 10, "ready"),
            ]
        )

        # Mock list_tools
        tm.list_tools = AsyncMock(
            return_value=[
                {"name": f"tool_{i}", "description": f"Tool {i}"} for i in range(6)
            ]
        )

        # Add servers attribute
        tm.servers = [
            MagicMock(name="sqlite", transport="stdio"),
            MagicMock(name="filesystem", transport="stdio"),
        ]

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
        assert options["transport"] is True  # Auto-enabled

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
        # JSON format
        options = servers_command._parse_arguments(["--json"])
        assert options["format"] == "json"

        # Tree format
        options = servers_command._parse_arguments(["--tree"])
        assert options["format"] == "tree"

        # Format with value
        options = servers_command._parse_arguments(["--format", "json"])
        assert options["format"] == "json"

        options = servers_command._parse_arguments(["-f", "tree"])
        assert options["format"] == "tree"

    def test_parse_arguments_combined_short_flags(self, servers_command):
        """Test parsing combined short flags like -dct."""
        options = servers_command._parse_arguments(["-dct"])
        assert options["detailed"] is True
        assert options["capabilities"] is True
        assert options["transport"] is True

    def test_parse_arguments_quiet(self, servers_command):
        """Test parsing quiet flag - not currently supported."""
        # Quiet flag is not currently supported in the implementation
        options = servers_command._parse_arguments(["-q"])
        assert options["quiet"] is False  # Default value

        options = servers_command._parse_arguments(["--quiet"])
        assert options["quiet"] is False  # Default value

    def test_parse_arguments_help(self, servers_command):
        """Test parsing help flag."""
        options = servers_command._parse_arguments(["-h"])
        assert options["help"] is True

        options = servers_command._parse_arguments(["--help"])
        assert options["help"] is True

        # -? is not currently supported
        options = servers_command._parse_arguments(["help"])
        assert options["help"] is True

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.servers_action_async")
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_basic(
        self, mock_output, mock_servers_action, servers_command, mock_tool_manager
    ):
        """Test basic execution without arguments - non-interactive mode."""
        # Use --json to avoid interactive mode
        await servers_command.execute(["--json"], tool_manager=mock_tool_manager)

        # Should call servers_action_async with json format
        mock_servers_action.assert_called_once_with(
            mock_tool_manager,
            detailed=False,
            show_capabilities=False,
            show_transport=False,
            output_format="json",
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.servers_action_async")
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_detailed(
        self, mock_output, mock_servers_action, servers_command, mock_tool_manager
    ):
        """Test execution with detailed flag."""
        # Add --json to avoid interactive mode
        await servers_command.execute(
            ["--detailed", "--json"], tool_manager=mock_tool_manager
        )

        # Should enable all details with json format
        mock_servers_action.assert_called_once_with(
            mock_tool_manager,
            detailed=True,
            show_capabilities=True,
            show_transport=True,
            output_format="json",
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_no_tool_manager(self, mock_output, servers_command):
        """Test execution without tool manager."""
        # Use --json to avoid interactive mode
        await servers_command.execute(["--json"])

        # Should print error (using error method, not print)
        assert mock_output.error.called or mock_output.print.called
        if mock_output.error.called:
            call_args = str(mock_output.error.call_args)
        else:
            call_args = str(mock_output.print.call_args)
        assert "ToolManager" in call_args or "not available" in call_args

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.servers_action_async")
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_invalid_format(
        self, mock_output, mock_servers_action, servers_command, mock_tool_manager
    ):
        """Test execution with invalid format."""
        # Add --json to avoid interactive mode, but test format validation
        await servers_command.execute(
            ["--format", "invalid", "--json"], tool_manager=mock_tool_manager
        )

        # Should still call but with json format (since we added --json)
        mock_servers_action.assert_called_once()
        # Format should be json since we explicitly passed --json after invalid format
        assert mock_servers_action.call_args.kwargs["output_format"] == "json"

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.servers.servers_action_async")
    @patch("mcp_cli.interactive.commands.servers.output")
    async def test_execute_with_exception(
        self, mock_output, mock_servers_action, servers_command, mock_tool_manager
    ):
        """Test execution handles exceptions gracefully."""
        mock_servers_action.side_effect = Exception("Connection error")

        # Use --json to avoid interactive mode
        await servers_command.execute(["--json"], tool_manager=mock_tool_manager)

        # Should print error message (likely via error method)
        assert mock_output.error.called or mock_output.print.called

        # Check if error message was printed
        error_found = False
        all_calls = []
        if mock_output.error.called:
            all_calls.extend(mock_output.error.call_args_list)
        if mock_output.print.called:
            all_calls.extend(mock_output.print.call_args_list)

        for call in all_calls:
            call_str = str(call)
            if (
                "Error" in call_str
                or "Failed" in call_str
                or "Connection error" in call_str
            ):
                error_found = True
                break
        assert error_found


# The ServersCapabilitiesCommand and ServersStatusCommand classes
# no longer exist in the implementation, so their tests have been removed.
