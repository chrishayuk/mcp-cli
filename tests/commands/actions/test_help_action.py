"""Tests for help action."""

from unittest.mock import MagicMock, patch
import pytest

from mcp_cli.commands.actions.help import (
    help_action,
    _get_commands,
    _show_command_help,
    _show_all_commands,
    _extract_description,
)


@pytest.fixture
def mock_commands():
    """Create mock command objects."""
    cmd1 = MagicMock()
    cmd1.name = "test_command"
    cmd1.help = "This is a test command\nUsage: test_command [args]"
    cmd1.aliases = ["tc", "test"]

    cmd2 = MagicMock()
    cmd2.name = "another_command"
    cmd2.help = "Another command description"
    cmd2.aliases = []

    return {"test_command": cmd1, "another_command": cmd2}


def test_extract_description():
    """Test _extract_description helper function."""
    # Test None input
    assert _extract_description(None) == "No description"

    # Test empty string
    assert _extract_description("") == "No description"

    # Test single line
    assert _extract_description("Simple description") == "Simple description"

    # Test multiline with usage
    help_text = """
    usage: command [args]
    This is the real description
    More details here
    """
    assert _extract_description(help_text) == "This is the real description"

    # Test multiline without usage
    help_text = "First line description\nSecond line details"
    assert _extract_description(help_text) == "First line description"

    # Test only usage line
    assert _extract_description("usage: command") == "No description"

    # Test whitespace handling
    assert _extract_description("   \n  Real description  \n   ") == "Real description"


def test_help_action_show_all_commands(mock_commands):
    """Test help_action showing all commands."""
    with (
        patch(
            "mcp_cli.commands.actions.help._get_commands", return_value=mock_commands
        ),
        patch("mcp_cli.commands.actions.help._show_all_commands") as mock_show_all,
    ):
        help_action()

        mock_show_all.assert_called_once_with(mock_commands)


def test_help_action_show_specific_command(mock_commands):
    """Test help_action showing specific command."""
    with (
        patch(
            "mcp_cli.commands.actions.help._get_commands", return_value=mock_commands
        ),
        patch("mcp_cli.commands.actions.help._show_command_help") as mock_show_cmd,
    ):
        help_action("test_command")

        mock_show_cmd.assert_called_once_with("test_command", mock_commands)


def test_help_action_with_console_param(mock_commands):
    """Test help_action with console parameter (backward compatibility)."""
    mock_console = MagicMock()

    with (
        patch(
            "mcp_cli.commands.actions.help._get_commands", return_value=mock_commands
        ),
        patch("mcp_cli.commands.actions.help._show_all_commands") as mock_show_all,
    ):
        help_action(console=mock_console)

        mock_show_all.assert_called_once_with(mock_commands)


def test_get_commands_interactive_registry():
    """Test _get_commands with InteractiveCommandRegistry."""
    mock_registry = MagicMock()
    mock_registry.get_all_commands.return_value = {"cmd1": "obj1", "cmd2": "obj2"}

    with patch("mcp_cli.commands.actions.help.Registry", return_value=mock_registry):
        result = _get_commands()

        assert result == {"cmd1": "obj1", "cmd2": "obj2"}


def test_get_commands_cli_registry_list():
    """Test _get_commands with CLI registry returning list."""
    mock_cmd1 = MagicMock()
    mock_cmd1.name = "cmd1"
    mock_cmd2 = MagicMock()
    mock_cmd2.name = "cmd2"

    mock_registry = MagicMock()
    mock_registry.get_all_commands.return_value = [mock_cmd1, mock_cmd2]

    with patch("mcp_cli.commands.actions.help.Registry", return_value=mock_registry):
        result = _get_commands()

        assert result == {"cmd1": mock_cmd1, "cmd2": mock_cmd2}


def test_get_commands_fallback_commands_attr():
    """Test _get_commands fallback to _commands attribute."""
    mock_registry = MagicMock()
    del mock_registry.get_all_commands  # Remove method
    mock_registry._commands = {"cmd1": "obj1"}

    with patch("mcp_cli.commands.actions.help.Registry", return_value=mock_registry):
        result = _get_commands()

        assert result == {"cmd1": "obj1"}


def test_get_commands_empty_fallback():
    """Test _get_commands when no commands available."""
    mock_registry = MagicMock()
    del mock_registry.get_all_commands
    del mock_registry._commands

    with patch("mcp_cli.commands.actions.help.Registry", return_value=mock_registry):
        result = _get_commands()

        assert result == {}


def test_show_command_help_existing_command():
    """Test _show_command_help with existing command."""
    mock_cmd = MagicMock()
    mock_cmd.name = "test_cmd"
    mock_cmd.help = "Test command help"
    mock_cmd.aliases = ["tc", "test"]

    commands = {"test_cmd": mock_cmd}

    with patch("mcp_cli.commands.actions.help.output") as mock_output:
        _show_command_help("test_cmd", commands)

        mock_output.panel.assert_called_once_with(
            "## test_cmd\n\nTest command help", title="Command Help", style="cyan"
        )
        mock_output.print.assert_called_once_with("\n[dim]Aliases: tc, test[/dim]")


def test_show_command_help_no_aliases():
    """Test _show_command_help with command that has no aliases."""
    mock_cmd = MagicMock()
    mock_cmd.name = "test_cmd"
    mock_cmd.help = "Test command help"
    mock_cmd.aliases = []

    commands = {"test_cmd": mock_cmd}

    with patch("mcp_cli.commands.actions.help.output") as mock_output:
        _show_command_help("test_cmd", commands)

        mock_output.panel.assert_called_once()
        mock_output.print.assert_not_called()  # No aliases to show


def test_show_command_help_missing_command():
    """Test _show_command_help with non-existent command."""
    commands = {}

    with patch("mcp_cli.commands.actions.help.output") as mock_output:
        _show_command_help("nonexistent", commands)

        mock_output.error.assert_called_once_with("Unknown command: nonexistent")


def test_show_all_commands(mock_commands):
    """Test _show_all_commands with mock commands."""
    with (
        patch("mcp_cli.commands.actions.help.output") as mock_output,
        patch("mcp_cli.commands.actions.help.format_table") as mock_format_table,
    ):
        mock_table = MagicMock()
        mock_format_table.return_value = mock_table

        _show_all_commands(mock_commands)

        # Verify table formatting
        mock_format_table.assert_called_once()
        table_data = mock_format_table.call_args[0][0]

        assert len(table_data) == 2
        assert table_data[0]["Command"] == "another_command"
        assert table_data[0]["Aliases"] == "-"
        assert table_data[1]["Command"] == "test_command"
        assert table_data[1]["Aliases"] == "tc, test"

        mock_output.print_table.assert_called_once_with(mock_table)
        mock_output.hint.assert_called_once_with(
            "\nType 'help <command>' for detailed information on a specific command."
        )


def test_show_all_commands_empty():
    """Test _show_all_commands with no commands."""
    with patch("mcp_cli.commands.actions.help.output") as mock_output:
        _show_all_commands({})

        mock_output.warning.assert_called_once_with("No commands available")


def test_show_command_help_missing_attributes():
    """Test _show_command_help with command missing optional attributes."""
    mock_cmd = MagicMock()
    # Simulate missing attributes
    mock_cmd.configure_mock(
        **{
            "name": "test_cmd",
            # No help attribute
            # No aliases attribute
        }
    )
    del mock_cmd.help
    del mock_cmd.aliases

    commands = {"test_cmd": mock_cmd}

    with patch("mcp_cli.commands.actions.help.output") as mock_output:
        _show_command_help("test_cmd", commands)

        # Should handle missing help gracefully
        mock_output.panel.assert_called_once_with(
            "## test_cmd\n\nNo description provided.",
            title="Command Help",
            style="cyan",
        )
        # Should not try to print aliases if they don't exist
        mock_output.print.assert_not_called()
