"""Tests for exit action."""

from unittest.mock import patch

from mcp_cli.commands.actions.exit import exit_action


def test_exit_action_interactive():
    """Test exit action in interactive mode."""
    with (
        patch("mcp_cli.commands.actions.exit.output") as mock_output,
        patch("mcp_cli.commands.actions.exit.restore_terminal") as mock_restore,
    ):
        result = exit_action(interactive=True)

        mock_output.info.assert_called_once_with("Exiting… Goodbye!")
        mock_restore.assert_called_once()
        assert result is True


def test_exit_action_non_interactive():
    """Test exit action in non-interactive mode."""
    with (
        patch("mcp_cli.commands.actions.exit.output") as mock_output,
        patch("mcp_cli.commands.actions.exit.restore_terminal") as mock_restore,
        patch("mcp_cli.commands.actions.exit.sys.exit") as mock_exit,
    ):
        # This should call sys.exit and not return
        exit_action(interactive=False)

        mock_output.info.assert_called_once_with("Exiting… Goodbye!")
        mock_restore.assert_called_once()
        mock_exit.assert_called_once_with(0)


def test_exit_action_default_interactive():
    """Test exit action with default interactive=True."""
    with (
        patch("mcp_cli.commands.actions.exit.output"),
        patch("mcp_cli.commands.actions.exit.restore_terminal"),
    ):
        result = exit_action()  # Default interactive=True
        assert result is True
