"""Tests for interactive mode theme command."""

from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

import pytest

from mcp_cli.interactive.commands.theme import ThemeCommand
from mcp_cli.utils.preferences import Theme


class TestInteractiveThemeCommand:
    """Test interactive mode theme command."""

    @pytest.fixture
    def theme_command(self):
        """Create theme command instance."""
        return ThemeCommand()

    @pytest.fixture
    def mock_shell(self):
        """Create mock interactive shell."""
        shell = MagicMock()
        shell.output = MagicMock()
        shell.output.print = MagicMock()
        shell.output.print_error = MagicMock()
        shell.output.print_success = MagicMock()
        shell.output.print_info = MagicMock()
        return shell

    def test_command_name(self, theme_command):
        """Test command name."""
        assert theme_command.name == "theme"

    def test_command_help(self, theme_command):
        """Test command help text."""
        assert "theme" in theme_command.help.lower()

    def test_command_aliases(self, theme_command):
        """Test command aliases."""
        assert theme_command.aliases == []

    @pytest.mark.asyncio
    async def test_execute_basic(self, theme_command):
        """Test basic execute functionality."""
        # The execute method is simple - it just calls theme_command
        # We'll test the theme_command function separately
        with patch("mcp_cli.interactive.commands.theme.theme_command") as mock_cmd:
            result = await theme_command.execute(["test"])
            mock_cmd.assert_called_once_with(["test"])
            assert result is None


class TestInteractiveThemeFunction:
    """Test the theme_command function directly."""
    
    @patch("mcp_cli.interactive.commands.theme.ask")
    @patch("mcp_cli.interactive.commands.theme.get_preference_manager")
    @patch("mcp_cli.interactive.commands.theme.set_theme")
    @patch("mcp_cli.interactive.commands.theme.output")
    def test_no_args_triggers_interactive(self, mock_output, mock_set_theme, mock_get_manager, mock_ask):
        """Test that no arguments triggers interactive selection."""
        from mcp_cli.interactive.commands.theme import theme_command
        
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"
        mock_get_manager.return_value = mock_manager
        # Mock selection returns theme number
        mock_ask.return_value = "2"  # dark is at index 2

        theme_command([])

        # Should trigger interactive selection
        mock_ask.assert_called_once()
        # Should set the selected theme
        mock_set_theme.assert_called_once_with("dark")
        mock_manager.set_theme.assert_called_once_with("dark")

    @patch("mcp_cli.interactive.commands.theme.get_preference_manager")
    @patch("mcp_cli.interactive.commands.theme.set_theme")
    @patch("mcp_cli.interactive.commands.theme.output")
    def test_show_current_and_list(self, mock_output, mock_set_theme, mock_get_manager):
        """Test showing current theme and listing with 'list' argument."""
        from mcp_cli.interactive.commands.theme import theme_command
        
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "minimal"
        mock_get_manager.return_value = mock_manager

        theme_command(["list"])

        mock_output.rule.assert_called_with("Theme Settings")
        # Check that current theme is displayed
        calls = str(mock_output.print.call_args_list)
        assert "minimal" in calls
        assert "Available themes:" in calls

    @patch("mcp_cli.interactive.commands.theme.show_theme_preview")
    @patch("mcp_cli.interactive.commands.theme.get_preference_manager")
    @patch("mcp_cli.interactive.commands.theme.set_theme")
    @patch("mcp_cli.interactive.commands.theme.output")
    def test_set_valid_theme(self, mock_output, mock_set_theme, mock_get_manager, mock_preview):
        """Test setting a valid theme."""
        from mcp_cli.interactive.commands.theme import theme_command
        
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        theme_command(["dark"])

        mock_set_theme.assert_called_once_with("dark")
        mock_manager.set_theme.assert_called_once_with("dark")
        mock_output.success.assert_called_with("Theme switched to: dark")

    @patch("mcp_cli.interactive.commands.theme.get_preference_manager")
    @patch("mcp_cli.interactive.commands.theme.set_theme")
    @patch("mcp_cli.interactive.commands.theme.output")
    def test_set_invalid_theme(self, mock_output, mock_set_theme, mock_get_manager):
        """Test setting an invalid theme."""
        from mcp_cli.interactive.commands.theme import theme_command
        
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        theme_command(["not_a_theme"])

        mock_set_theme.assert_not_called()
        mock_manager.set_theme.assert_not_called()
        mock_output.error.assert_called()
        error_msg = str(mock_output.error.call_args)
        assert "Invalid theme: not_a_theme" in error_msg

    @patch("mcp_cli.interactive.commands.theme.show_theme_preview")
    @patch("mcp_cli.interactive.commands.theme.get_preference_manager")
    @patch("mcp_cli.interactive.commands.theme.ask")
    @patch("mcp_cli.interactive.commands.theme.set_theme")
    @patch("mcp_cli.interactive.commands.theme.output")
    def test_select_theme_interactive(
        self, mock_output, mock_set_theme, mock_ask, mock_get_manager, mock_preview
    ):
        """Test interactive theme selection."""
        from mcp_cli.interactive.commands.theme import theme_command
        
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"
        mock_get_manager.return_value = mock_manager
        
        # Mock selection to return theme number
        mock_ask.return_value = "4"  # minimal is at index 4

        theme_command(["select"])

        # Should call selection
        mock_ask.assert_called_once()
        
        # Check that theme was set
        mock_set_theme.assert_called_once_with("minimal")
        mock_manager.set_theme.assert_called_once_with("minimal")

    @patch("mcp_cli.interactive.commands.theme.get_preference_manager")
    @patch("mcp_cli.interactive.commands.theme.ask")
    @patch("mcp_cli.interactive.commands.theme.set_theme")
    @patch("mcp_cli.interactive.commands.theme.output")
    def test_select_theme_cancelled(self, mock_output, mock_set_theme, mock_ask, mock_get_manager):
        """Test invalid number in theme selection."""
        from mcp_cli.interactive.commands.theme import theme_command
        
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"
        mock_get_manager.return_value = mock_manager
        mock_ask.return_value = "99"  # Invalid number

        theme_command(["select"])

        # Should not change anything with invalid number
        mock_manager.set_theme.assert_not_called()
        mock_set_theme.assert_not_called()
        
        # Should show error
        mock_output.error.assert_called()