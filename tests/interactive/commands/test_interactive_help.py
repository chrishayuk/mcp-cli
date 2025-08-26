"""Tests for interactive mode help command."""

from unittest.mock import MagicMock, patch
import pytest

from mcp_cli.interactive.commands.help import HelpCommand
from mcp_cli.interactive.registry import InteractiveCommandRegistry


class DummyCommand:
    """Dummy command for testing."""

    def __init__(self, name, help_text="", aliases=None):
        self.name = name
        self.help = help_text
        self.aliases = aliases or []

    async def execute(self, args, tool_manager=None, **kwargs):
        """Dummy execute method."""
        return True


class TestInteractiveHelpCommand:
    """Test interactive mode help command."""

    @pytest.fixture
    def help_command(self):
        """Create HelpCommand instance."""
        return HelpCommand()

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Set up command registry for tests."""
        # Store original state
        original_commands = InteractiveCommandRegistry._commands.copy()
        original_aliases = InteractiveCommandRegistry._aliases.copy()

        # Clear and add test commands
        InteractiveCommandRegistry._commands.clear()
        InteractiveCommandRegistry._aliases.clear()

        # Register test commands
        test_cmd = DummyCommand("test", "Test command for testing", ["t"])
        another_cmd = DummyCommand(
            "another", "Another test command with longer help text"
        )
        InteractiveCommandRegistry.register(test_cmd)
        InteractiveCommandRegistry.register(another_cmd)

        yield

        # Restore original state
        InteractiveCommandRegistry._commands = original_commands
        InteractiveCommandRegistry._aliases = original_aliases

    def test_command_properties(self, help_command):
        """Test HelpCommand basic properties."""
        assert help_command.name == "help"
        assert "?" in help_command.aliases or "h" in help_command.aliases
        assert "help" in help_command.help.lower()

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.help.help_action")
    async def test_execute_no_arguments(self, mock_help_action, help_command):
        """Test help with no arguments shows all commands."""
        result = await help_command.execute([], tool_manager=None)

        # Should call help_action with None (show all)
        mock_help_action.assert_called_once_with(None)
        assert result is None  # Doesn't exit interactive mode

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.help.help_action")
    async def test_execute_with_command_name(self, mock_help_action, help_command):
        """Test help with specific command name."""
        result = await help_command.execute(["test"], tool_manager=None)

        # Should call help_action with the command name
        mock_help_action.assert_called_once_with("test")
        assert result is None

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.help.help_action")
    async def test_execute_with_slash_command(self, mock_help_action, help_command):
        """Test help handles command names with leading slash."""
        result = await help_command.execute(["/test"], tool_manager=None)

        # Should strip the slash and call help_action
        mock_help_action.assert_called_once_with("test")
        assert result is None

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.help.help_action")
    async def test_execute_with_multiple_args(self, mock_help_action, help_command):
        """Test help with multiple arguments only uses the first."""
        result = await help_command.execute(
            ["test", "extra", "args"], tool_manager=None
        )

        # Should only use the first argument
        mock_help_action.assert_called_once_with("test")
        assert result is None

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.help.format_table")
    @patch("mcp_cli.commands.help.output")
    async def test_help_action_integration(
        self, mock_output, mock_format_table, help_command
    ):
        """Test that help command integrates with help_action properly."""
        # Mock format_table to prevent actual table creation
        mock_format_table.return_value = MagicMock()

        # This tests the actual integration without mocking help_action
        result = await help_command.execute([], tool_manager=None)

        # Should have called output methods or format_table
        assert (
            mock_format_table.called
            or mock_output.warning.called
            or mock_output.print_table.called
        )
        assert result is None

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.help.format_table")
    @patch("mcp_cli.commands.help.output")
    async def test_show_all_commands(
        self, mock_output, mock_format_table, help_command
    ):
        """Test showing all commands creates a table."""
        mock_format_table.return_value = MagicMock()

        result = await help_command.execute([], tool_manager=None)

        # Should format a table with commands
        mock_format_table.assert_called_once()
        table_data = mock_format_table.call_args[0][0]

        # Should have our test commands
        command_names = [row["Command"] for row in table_data]
        assert "test" in command_names
        assert "another" in command_names

        assert result is None

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.help.output")
    async def test_show_specific_command(self, mock_output, help_command):
        """Test showing help for a specific command."""
        result = await help_command.execute(["test"], tool_manager=None)

        # Should display panel with command details
        mock_output.panel.assert_called_once()
        panel_content = mock_output.panel.call_args[0][0]

        assert "test" in panel_content.lower()
        assert result is None

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.help.output")
    async def test_unknown_command(self, mock_output, help_command):
        """Test help with unknown command shows error."""
        result = await help_command.execute(["unknown"], tool_manager=None)

        # Should show error
        mock_output.error.assert_called_once()
        error_msg = mock_output.error.call_args[0][0]
        assert "unknown" in error_msg.lower()
        assert result is None

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.help.output")
    async def test_empty_registry(self, mock_output, help_command):
        """Test help with empty command registry."""
        # Clear all commands
        InteractiveCommandRegistry._commands.clear()

        result = await help_command.execute([], tool_manager=None)

        # Should show warning about no commands
        mock_output.warning.assert_called_once()
        warning_msg = mock_output.warning.call_args[0][0]
        assert "no commands" in warning_msg.lower()
        assert result is None

    @pytest.mark.asyncio
    @patch("mcp_cli.commands.help.output")
    async def test_command_with_aliases(self, mock_output, help_command):
        """Test help shows command aliases."""
        result = await help_command.execute(["test"], tool_manager=None)

        # Should show aliases
        mock_output.panel.assert_called_once()

        # Check if aliases are mentioned
        if mock_output.print.called:
            print_calls = str(mock_output.print.call_args_list)
            # Test command has alias 't'
            assert "t" in print_calls or "Aliases" in print_calls

        assert result is None

    @pytest.mark.asyncio
    @patch("mcp_cli.interactive.commands.help.help_action")
    async def test_help_strips_slash_prefix(self, mock_help_action, help_command):
        """Test that help strips '/' prefix from command names."""
        # Test various forms
        test_cases = [
            (["/foo"], "foo"),
            (["//foo"], "foo"),  # lstrip removes all leading slashes
            (["foo"], "foo"),  # No slash stays the same
        ]

        for args, expected in test_cases:
            mock_help_action.reset_mock()
            await help_command.execute(args, tool_manager=None)
            mock_help_action.assert_called_once_with(expected)

    @pytest.mark.asyncio
    async def test_help_command_registration(self):
        """Test that help command is properly registered."""
        # Register the help command if not already registered
        help_cmd = HelpCommand()
        InteractiveCommandRegistry.register(help_cmd)

        # The help command should be in the registry
        cmd = InteractiveCommandRegistry.get_command("help")
        assert cmd is not None
        assert isinstance(cmd, HelpCommand)

        # Check aliases work
        cmd_from_alias = InteractiveCommandRegistry.get_command("?")
        assert cmd_from_alias is not None
        # They should be the same command
        assert cmd_from_alias.name == cmd.name
