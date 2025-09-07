"""Tests for chat mode theme command."""

from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.chat.commands.theme import (
    handle_theme_command,
    show_theme_table,
    get_theme_info,
    show_theme_preview,
    show_theme_card,
    interactive_theme_selection,
)


class TestChatThemeCommand:
    """Test chat mode theme command."""

    @pytest.fixture
    def mock_context(self):
        """Create mock chat context."""
        context = MagicMock()
        return context

    @pytest.mark.asyncio
    @patch("chuk_term.ui.prompts.confirm")
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.get_theme")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_show_current_theme(
        self,
        mock_get_manager,
        mock_set_theme,
        mock_output,
        mock_get_theme,
        mock_ask,
        mock_confirm,
        mock_context,
    ):
        """Test showing current theme with no arguments triggers interactive selection."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "dark"
        mock_get_manager.return_value = mock_manager
        mock_get_theme.return_value = "dark"
        # Mock selection returns theme number for monokai
        mock_ask.return_value = "6"  # monokai is at index 6
        mock_confirm.return_value = True  # Confirm the theme change

        await handle_theme_command(mock_context, [])

        # Should trigger interactive selection
        mock_ask.assert_called_once()
        # Should apply theme multiple times for preview and final
        # Check that monokai was set at some point
        theme_calls = [call[0][0] for call in mock_set_theme.call_args_list]
        assert "monokai" in theme_calls
        mock_manager.set_theme.assert_called_once_with("monokai")

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_set_valid_theme(
        self, mock_get_manager, mock_set_theme, mock_output, mock_context
    ):
        """Test setting a valid theme."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"  # Current theme
        mock_get_manager.return_value = mock_manager

        await handle_theme_command(mock_context, ["monokai"])

        # Should set theme in UI and preferences
        mock_set_theme.assert_called_with("monokai")
        mock_manager.set_theme.assert_called_once_with("monokai")

        # Should display success message (could be in preview)
        assert mock_output.success.called
        success_calls = str(mock_output.success.call_args_list)
        assert "monokai" in success_calls or "Theme switched" in success_calls

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_set_invalid_theme(
        self, mock_get_manager, mock_set_theme, mock_output, mock_context
    ):
        """Test setting an invalid theme."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        await handle_theme_command(mock_context, ["invalid_theme"])

        # Should not set theme
        mock_set_theme.assert_not_called()
        mock_manager.set_theme.assert_not_called()

        # Should display error message
        mock_output.error.assert_called()
        error_args = str(mock_output.error.call_args)
        assert "Invalid theme" in error_args
        assert "invalid_theme" in error_args

    @pytest.mark.asyncio
    @patch("chuk_term.ui.prompts.confirm")
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.get_theme")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_select_theme_interactive(
        self,
        mock_get_manager,
        mock_set_theme,
        mock_output,
        mock_get_theme,
        mock_ask,
        mock_confirm,
        mock_context,
    ):
        """Test interactive theme selection."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"
        mock_get_manager.return_value = mock_manager
        mock_get_theme.return_value = "default"

        # Mock the selection to return theme number
        mock_ask.return_value = "6"  # monokai is at index 6
        mock_confirm.return_value = True  # Confirm the theme change

        await handle_theme_command(mock_context, [])

        # Should prompt for selection
        mock_ask.assert_called_once()
        prompt = mock_ask.call_args[0][0]
        assert "Enter theme number" in prompt
        assert (
            mock_ask.call_args.kwargs.get("default") == ""
        )  # empty default for keeping current

        # Should set selected theme (monokai) - check it was called with monokai
        theme_calls = [call[0][0] for call in mock_set_theme.call_args_list]
        assert "monokai" in theme_calls
        mock_manager.set_theme.assert_called_once_with("monokai")

        # Should display success
        assert mock_output.success.called

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.format_table")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_list_themes(
        self, mock_get_manager, mock_format_table, mock_output, mock_context
    ):
        """Test listing all available themes."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"
        mock_get_manager.return_value = mock_manager
        mock_format_table.return_value = "formatted table"

        await handle_theme_command(mock_context, ["list"])

        # Should create and display a table
        mock_format_table.assert_called_once()
        call_kwargs = mock_format_table.call_args.kwargs
        assert call_kwargs["title"] == "Available Themes"
        assert "data" in call_kwargs
        assert len(call_kwargs["data"]) > 0

        # Should print the table
        mock_output.print.assert_any_call("formatted table")

    @pytest.mark.asyncio
    @patch("chuk_term.ui.prompts.confirm")
    @patch("mcp_cli.chat.commands.theme.get_theme")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_preview_theme(
        self,
        mock_get_manager,
        mock_set_theme,
        mock_output,
        mock_get_theme,
        mock_confirm,
        mock_context,
    ):
        """Test previewing a theme without applying."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"
        mock_get_manager.return_value = mock_manager
        mock_get_theme.return_value = "default"
        mock_confirm.return_value = False  # Don't keep the theme

        await handle_theme_command(mock_context, ["preview", "dark"])

        # Should temporarily apply the theme for preview
        theme_calls = [call[0][0] for call in mock_set_theme.call_args_list]
        assert "dark" in theme_calls
        assert "default" in theme_calls  # Should revert back

        # Should not save the theme
        mock_manager.set_theme.assert_not_called()

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_preview_invalid_theme(
        self, mock_get_manager, mock_output, mock_context
    ):
        """Test previewing an invalid theme."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        await handle_theme_command(mock_context, ["preview", "nonexistent"])

        # Should display error
        mock_output.error.assert_called()
        error_args = str(mock_output.error.call_args)
        assert "Invalid theme" in error_args
        assert "nonexistent" in error_args

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_preview_without_theme_name(
        self, mock_get_manager, mock_output, mock_context
    ):
        """Test preview command without theme name."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        await handle_theme_command(mock_context, ["preview"])

        # Should display error about usage
        mock_output.error.assert_called_with("Usage: /theme preview <name>")

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_set_same_theme(
        self, mock_get_manager, mock_set_theme, mock_output, mock_context
    ):
        """Test setting the same theme that's already active."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "dark"
        mock_get_manager.return_value = mock_manager

        await handle_theme_command(mock_context, ["dark"])

        # Should not set theme
        mock_set_theme.assert_not_called()
        mock_manager.set_theme.assert_not_called()

        # Should inform user
        mock_output.info.assert_called()
        info_args = str(mock_output.info.call_args)
        assert "Already using dark theme" in info_args

    def test_get_theme_info_known_theme(self):
        """Test getting theme info for a known theme."""
        info = get_theme_info("dark")
        assert info["name"] == "Dark"
        assert "Dark mode" in info["description"]
        assert "Low-light" in info["best_for"]

    def test_get_theme_info_unknown_theme(self):
        """Test getting theme info for an unknown theme."""
        info = get_theme_info("custom_theme")
        assert info["name"] == "Custom_Theme"
        assert "Theme: custom_theme" in info["description"]
        assert info["best_for"] == "Custom theme"

    @patch("mcp_cli.chat.commands.theme.output")
    def test_show_theme_preview(self, mock_output):
        """Test showing theme preview."""
        show_theme_preview("test_theme")

        # Should display various message types
        mock_output.info.assert_called()
        mock_output.success.assert_called()
        mock_output.warning.assert_called()
        mock_output.error.assert_called()
        mock_output.hint.assert_called()

    @patch("mcp_cli.chat.commands.theme.output")
    def test_show_theme_card(self, mock_output):
        """Test showing theme card."""
        show_theme_card("dark", is_current=True)

        # Should show rule and theme details
        mock_output.rule.assert_called()
        assert (
            mock_output.print.call_count >= 4
        )  # Description, best for, style, preview

    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.format_table")
    def test_show_theme_table(self, mock_format_table, mock_output):
        """Test showing theme table."""
        mock_format_table.return_value = "formatted table"

        show_theme_table("default")

        # Should format and print table
        mock_format_table.assert_called_once()
        call_kwargs = mock_format_table.call_args.kwargs
        assert "data" in call_kwargs
        assert call_kwargs["title"] == "Available Themes"

        mock_output.print.assert_any_call("formatted table")

    @pytest.mark.asyncio
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_interactive_empty_response(
        self, mock_get_manager, mock_output, mock_ask, mock_context
    ):
        """Test interactive selection with empty response (keep current)."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"
        mock_get_manager.return_value = mock_manager

        # Mock empty response to keep current
        mock_ask.return_value = ""

        await interactive_theme_selection(mock_context, mock_manager)

        # Should not change theme
        mock_manager.set_theme.assert_not_called()

        # Should inform about keeping current
        mock_output.info.assert_called()
        info_args = str(mock_output.info.call_args_list)
        assert "Keeping current theme" in info_args

    @pytest.mark.asyncio
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_interactive_invalid_number(
        self, mock_get_manager, mock_output, mock_ask, mock_context
    ):
        """Test interactive selection with invalid number."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"
        mock_get_manager.return_value = mock_manager

        # Mock invalid number
        mock_ask.return_value = "99"

        await interactive_theme_selection(mock_context, mock_manager)

        # Should not change theme
        mock_manager.set_theme.assert_not_called()

        # Should show error
        mock_output.error.assert_called()
        error_args = str(mock_output.error.call_args)
        assert "Invalid number: 99" in error_args

    @pytest.mark.asyncio
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_interactive_invalid_name(
        self, mock_get_manager, mock_output, mock_ask, mock_context
    ):
        """Test interactive selection with invalid theme name."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"
        mock_get_manager.return_value = mock_manager

        # Mock invalid theme name
        mock_ask.return_value = "nonexistent"

        await interactive_theme_selection(mock_context, mock_manager)

        # Should not change theme
        mock_manager.set_theme.assert_not_called()

        # Should show error
        mock_output.error.assert_called()
        error_args = str(mock_output.error.call_args)
        assert "Unknown theme: nonexistent" in error_args
