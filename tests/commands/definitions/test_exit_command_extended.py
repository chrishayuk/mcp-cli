"""Extended tests for exit command to achieve 100% coverage."""

import pytest
from mcp_cli.commands.definitions.exit import ExitCommand
from mcp_cli.commands.base import CommandMode


@pytest.fixture
def exit_command():
    """Create an exit command instance."""
    return ExitCommand()


def test_exit_help_text(exit_command):
    """Test the help_text property."""
    help_text = exit_command.help_text
    assert "Exit the current session" in help_text
    assert "/exit" in help_text
    assert "Aliases: quit, q, bye" in help_text
    assert "CLI mode" in help_text


def test_exit_modes_property(exit_command):
    """Test the modes property."""
    modes = exit_command.modes
    assert modes == (CommandMode.CHAT | CommandMode.INTERACTIVE)
    # Verify it's not available in CLI mode
    assert not (modes & CommandMode.CLI)