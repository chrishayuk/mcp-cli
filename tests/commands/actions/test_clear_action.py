"""Tests for clear action."""

from unittest.mock import patch
from mcp_cli.commands.actions.clear import clear_action


def test_clear_action_basic():
    """Test basic clear action without verbose."""
    with patch("mcp_cli.commands.actions.clear.clear_screen") as mock_clear:
        clear_action()
        mock_clear.assert_called_once()


def test_clear_action_verbose():
    """Test clear action with verbose output."""
    with (
        patch("mcp_cli.commands.actions.clear.clear_screen") as mock_clear,
        patch("mcp_cli.commands.actions.clear.output") as mock_output,
    ):
        clear_action(verbose=True)
        mock_clear.assert_called_once()
        mock_output.hint.assert_called_once_with("Screen cleared.")


def test_clear_action_verbose_false():
    """Test clear action with verbose=False doesn't output."""
    with (
        patch("mcp_cli.commands.actions.clear.clear_screen") as mock_clear,
        patch("mcp_cli.commands.actions.clear.output") as mock_output,
    ):
        clear_action(verbose=False)
        mock_clear.assert_called_once()
        mock_output.hint.assert_not_called()
