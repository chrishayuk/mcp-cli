"""Tests for the theme commands."""

import pytest
from unittest.mock import patch, Mock
from mcp_cli.commands.definitions.theme_singular import ThemeSingularCommand
from mcp_cli.commands.definitions.themes_plural import ThemesPluralCommand


class TestThemeSingularCommand:
    """Test the ThemeSingularCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ThemeSingularCommand instance."""
        return ThemeSingularCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "theme"
        assert command.aliases == []
        assert "current theme" in command.description.lower()
        assert command.requires_context is True  # Uses default

    @pytest.mark.asyncio
    async def test_execute_show_current(self, command):
        """Test showing current theme."""
        # Just test that the command executes without error
        # The actual theme display is handled by chuk_term which is tested separately
        with patch("chuk_term.ui") as mock_ui:
            # Mock the entire ui module to avoid import issues
            mock_ui.output = Mock()
            mock_ui.theme = Mock()
            mock_ui.theme.get_theme.return_value = Mock(
                name="dark", description="Dark theme"
            )

            result = await command.execute()

            # We just care that it executes successfully
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_theme(self, command):
        """Test setting a theme."""
        with patch("mcp_cli.commands.actions.theme.theme_action_async") as mock_action:
            result = await command.execute(args=["dark"])

            mock_action.assert_called_once()
            call_args = mock_action.call_args[0][0]
            assert call_args.theme_name == "dark"
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_invalid_theme(self, command):
        """Test setting an invalid theme."""
        with patch("mcp_cli.commands.actions.theme.theme_action_async") as mock_action:
            mock_action.side_effect = ValueError("Invalid theme")

            result = await command.execute(args=["invalid"])

            # The command will catch the exception and return an error
            assert result.success is False


class TestThemesPluralCommand:
    """Test the ThemesPluralCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ThemesPluralCommand instance."""
        return ThemesPluralCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "themes"
        assert command.aliases == []
        assert "List all available themes" in command.description
        assert command.requires_context is True  # Uses default

    @pytest.mark.asyncio
    async def test_execute_list_themes(self, command):
        """Test listing available themes."""
        with patch("mcp_cli.commands.actions.theme.theme_action_async") as mock_action:
            # Mock the theme action
            mock_action.return_value = (
                None  # theme_action_async doesn't return anything
            )

            result = await command.execute()

            assert result.success is True
            mock_action.assert_called_once()
            call_args = mock_action.call_args[0][0]
            # For themes plural, it shows list (no specific theme_name)
            assert call_args.theme_name is None or call_args.theme_name == ""
