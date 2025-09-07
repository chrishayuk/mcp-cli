"""Comprehensive tests for the theme command with full coverage."""

import pytest
from unittest.mock import MagicMock, patch
from rich.table import Table

from mcp_cli.chat.commands.theme import (
    cmd_theme,
    handle_theme_command,
    interactive_theme_selection,
)


class TestThemeCommandComprehensive:
    """Comprehensive tests for theme command functionality."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock ApplicationContext."""
        from mcp_cli.context import ApplicationContext

        context = MagicMock(spec=ApplicationContext)
        context.provider = "openai"
        context.model = "gpt-4"
        context.verbose_mode = True
        context._extra = {}
        return context

    @pytest.fixture
    def mock_pref_manager(self):
        """Create a mock preference manager."""
        manager = MagicMock()
        manager.get_theme.return_value = "default"
        manager.set_theme = MagicMock()
        return manager

    # ===== Command Entry Point Tests =====

    @pytest.mark.asyncio
    @patch("mcp_cli.context.get_context")
    @patch("mcp_cli.chat.commands.theme.handle_theme_command")
    async def test_cmd_theme_entry_point(
        self, mock_handle, mock_get_context, mock_context
    ):
        """Test the main /theme command entry point."""
        mock_get_context.return_value = mock_context

        # Test with no arguments
        result = await cmd_theme(["/theme"])
        assert result is True
        mock_handle.assert_called_once_with(mock_context, [])

        # Test with theme name
        mock_handle.reset_mock()
        result = await cmd_theme(["/theme", "dark"])
        assert result is True
        mock_handle.assert_called_once_with(mock_context, ["dark"])

    # ===== Direct Theme Setting Tests =====

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.output")
    async def test_set_theme_directly_valid(
        self,
        mock_output,
        mock_set_theme,
        mock_get_pref,
        mock_context,
        mock_pref_manager,
    ):
        """Test setting a valid theme directly."""
        mock_get_pref.return_value = mock_pref_manager
        # Set a different current theme so all themes can be switched to
        mock_pref_manager.get_theme.return_value = "minimal"

        # Test each valid theme (except minimal which is current)
        valid_themes = [
            "default",
            "dark",
            "light",
            "terminal",
            "monokai",
            "dracula",
            "solarized",
        ]

        for theme in valid_themes:
            mock_set_theme.reset_mock()
            mock_pref_manager.set_theme.reset_mock()
            mock_output.reset_mock()

            await handle_theme_command(mock_context, [theme])

            # Should apply and save theme
            mock_set_theme.assert_called_once_with(theme)
            mock_pref_manager.set_theme.assert_called_once_with(theme)
            mock_output.success.assert_called()
            assert theme in str(mock_output.success.call_args)

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.output")
    async def test_set_theme_directly_invalid(
        self,
        mock_output,
        mock_set_theme,
        mock_get_pref,
        mock_context,
        mock_pref_manager,
    ):
        """Test setting an invalid theme directly."""
        mock_get_pref.return_value = mock_pref_manager

        await handle_theme_command(mock_context, ["not_a_theme"])

        # Should not apply theme
        mock_set_theme.assert_not_called()
        mock_pref_manager.set_theme.assert_not_called()

        # Should show error
        mock_output.error.assert_called()
        assert "Invalid theme" in str(mock_output.error.call_args)
        mock_output.hint.assert_called()

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.output")
    async def test_set_theme_error_handling(
        self,
        mock_output,
        mock_set_theme,
        mock_get_pref,
        mock_context,
        mock_pref_manager,
    ):
        """Test error handling when setting theme fails."""
        mock_get_pref.return_value = mock_pref_manager
        mock_set_theme.side_effect = Exception("Theme error")

        await handle_theme_command(mock_context, ["dark"])

        # Should show error message
        mock_output.error.assert_called()
        assert "Failed to switch theme" in str(mock_output.error.call_args)

    # ===== Interactive Selection Tests =====

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    @patch("mcp_cli.chat.commands.theme.interactive_theme_selection")
    async def test_interactive_mode_no_args(
        self, mock_interactive, mock_get_pref, mock_context, mock_pref_manager
    ):
        """Test that no arguments triggers interactive mode."""
        mock_get_pref.return_value = mock_pref_manager

        await handle_theme_command(mock_context, [])

        mock_interactive.assert_called_once_with(mock_context, mock_pref_manager)

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    @patch("mcp_cli.chat.commands.theme.output")
    async def test_invalid_subcommand(
        self, mock_output, mock_get_pref, mock_context, mock_pref_manager
    ):
        """Test that invalid subcommand shows error."""
        mock_get_pref.return_value = mock_pref_manager

        await handle_theme_command(mock_context, ["select"])

        # "select" is no longer a special command, treated as invalid theme name
        mock_output.error.assert_called()
        assert "Invalid theme" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_theme")
    @patch("chuk_term.ui.prompts.confirm")
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    async def test_interactive_selection_by_number(
        self,
        mock_set_theme,
        mock_output,
        mock_ask,
        mock_confirm,
        mock_get_theme,
        mock_context,
        mock_pref_manager,
    ):
        """Test selecting a theme by number in interactive mode."""
        mock_pref_manager.get_theme.return_value = "default"
        mock_get_theme.return_value = (
            "default"  # For theme restoration during comparison
        )
        mock_confirm.return_value = True  # Always confirm changes

        # Test selecting each theme by number
        for i in range(1, 9):
            mock_ask.return_value = str(i)
            mock_set_theme.reset_mock()
            mock_pref_manager.set_theme.reset_mock()

            await interactive_theme_selection(mock_context, mock_pref_manager)

            # Should set the corresponding theme
            themes = [
                "default",
                "dark",
                "light",
                "minimal",
                "terminal",
                "monokai",
                "dracula",
                "solarized",
            ]
            expected_theme = themes[i - 1]

            if expected_theme != "default":  # Won't switch if already current
                # set_theme is called multiple times for preview, so just check it was called with expected
                assert any(
                    call[0][0] == expected_theme
                    for call in mock_set_theme.call_args_list
                )
                mock_pref_manager.set_theme.assert_called_once_with(expected_theme)

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_theme")
    @patch("chuk_term.ui.prompts.confirm")
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    async def test_interactive_selection_by_name(
        self,
        mock_set_theme,
        mock_output,
        mock_ask,
        mock_confirm,
        mock_get_theme,
        mock_context,
        mock_pref_manager,
    ):
        """Test selecting a theme by name in interactive mode."""
        mock_pref_manager.get_theme.return_value = "default"
        mock_get_theme.return_value = "default"  # For theme restoration
        mock_confirm.return_value = True  # Always confirm changes

        # Test selecting by name
        mock_ask.return_value = "monokai"

        await interactive_theme_selection(mock_context, mock_pref_manager)

        # set_theme is called multiple times for preview, so just check it was called with expected
        assert any(call[0][0] == "monokai" for call in mock_set_theme.call_args_list)
        mock_pref_manager.set_theme.assert_called_once_with("monokai")

    @pytest.mark.asyncio
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    async def test_interactive_selection_invalid_number(
        self, mock_set_theme, mock_output, mock_ask, mock_context, mock_pref_manager
    ):
        """Test selecting an invalid number in interactive mode."""
        mock_pref_manager.get_theme.return_value = "default"
        mock_ask.return_value = "99"  # Invalid number

        await interactive_theme_selection(mock_context, mock_pref_manager)

        # Should not set any theme
        mock_set_theme.assert_not_called()
        mock_pref_manager.set_theme.assert_not_called()

        # Should show error
        mock_output.error.assert_called()
        assert "Invalid number" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    async def test_interactive_selection_invalid_name(
        self, mock_set_theme, mock_output, mock_ask, mock_context, mock_pref_manager
    ):
        """Test selecting an invalid name in interactive mode."""
        mock_pref_manager.get_theme.return_value = "default"
        mock_ask.return_value = "not_a_theme"

        await interactive_theme_selection(mock_context, mock_pref_manager)

        # Should not set any theme
        mock_set_theme.assert_not_called()
        mock_pref_manager.set_theme.assert_not_called()

        # Should show error
        mock_output.error.assert_called()
        assert "Unknown theme" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    async def test_interactive_selection_same_theme(
        self, mock_set_theme, mock_output, mock_ask, mock_context, mock_pref_manager
    ):
        """Test selecting the current theme in interactive mode."""
        mock_pref_manager.get_theme.return_value = "dark"
        mock_ask.return_value = "2"  # dark is at index 2

        await interactive_theme_selection(mock_context, mock_pref_manager)

        # Should not set theme (already current)
        mock_set_theme.assert_not_called()
        mock_pref_manager.set_theme.assert_not_called()

        # Should show info message
        mock_output.info.assert_called()
        assert "Already using" in str(mock_output.info.call_args)

    # ===== UI Display Tests =====

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_theme")
    @patch("chuk_term.ui.prompts.confirm")
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    async def test_interactive_gallery_display(
        self,
        mock_set_theme,
        mock_output,
        mock_ask,
        mock_confirm,
        mock_get_theme,
        mock_context,
        mock_pref_manager,
    ):
        """Test that the theme gallery is displayed correctly."""
        mock_pref_manager.get_theme.return_value = "monokai"
        mock_get_theme.return_value = "monokai"
        mock_ask.return_value = "1"  # Select default
        mock_confirm.return_value = True  # Confirm change

        await interactive_theme_selection(mock_context, mock_pref_manager)

        # Should display the theme selector rule
        mock_output.rule.assert_any_call("Theme Selector")

        # Should print table
        mock_output.print.assert_called()

        # Check that a Table was created and printed
        printed_items = [
            call[0][0] for call in mock_output.print.call_args_list if call[0]
        ]
        tables = [item for item in printed_items if isinstance(item, Table)]

        assert len(tables) > 0, "No table was printed"
        table = tables[0]

        # Verify table structure
        assert table.title == "Available Themes"

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_theme")
    @patch("chuk_term.ui.prompts.confirm")
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    async def test_theme_preview_display(
        self,
        mock_set_theme,
        mock_output,
        mock_ask,
        mock_confirm,
        mock_get_theme,
        mock_context,
        mock_pref_manager,
    ):
        """Test that theme preview is shown after selection."""
        mock_pref_manager.get_theme.return_value = "default"
        mock_get_theme.return_value = "default"
        mock_ask.return_value = "6"  # monokai
        mock_confirm.return_value = True  # Confirm change

        await interactive_theme_selection(mock_context, mock_pref_manager)

        # Should show preview messages
        output_calls = mock_output.method_calls
        method_names = [call[0] for call in output_calls]

        # Should show various preview elements
        assert "info" in method_names
        assert "success" in method_names
        assert "warning" in method_names
        assert "error" in method_names
        assert "hint" in method_names

    # ===== Case Sensitivity Tests =====

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.output")
    async def test_case_insensitive_theme_names(
        self,
        mock_output,
        mock_set_theme,
        mock_get_pref,
        mock_context,
        mock_pref_manager,
    ):
        """Test that theme names are case-insensitive."""
        mock_get_pref.return_value = mock_pref_manager

        # Test various case combinations
        test_cases = ["DARK", "Dark", "dArK", "MoNoKaI", "DRACULA"]
        expected = ["dark", "dark", "dark", "monokai", "dracula"]

        for test_theme, expected_theme in zip(test_cases, expected):
            mock_set_theme.reset_mock()
            mock_pref_manager.set_theme.reset_mock()

            await handle_theme_command(mock_context, [test_theme])

            mock_set_theme.assert_called_once_with(expected_theme)
            mock_pref_manager.set_theme.assert_called_once_with(expected_theme)

    # ===== New Command Tests =====

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    @patch("mcp_cli.chat.commands.theme.output")
    async def test_list_command(
        self, mock_output, mock_get_pref, mock_context, mock_pref_manager
    ):
        """Test the /theme list command."""
        mock_get_pref.return_value = mock_pref_manager
        mock_pref_manager.get_theme.return_value = "dark"

        await handle_theme_command(mock_context, ["list"])

        # Should display themes header
        mock_output.rule.assert_any_call("Available Themes")
        # Should display theme information
        mock_output.print.assert_called()

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    @patch("mcp_cli.chat.commands.theme.get_theme")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("chuk_term.ui.prompts.confirm")
    @patch("mcp_cli.chat.commands.theme.output")
    async def test_preview_command(
        self,
        mock_output,
        mock_confirm,
        mock_set_theme,
        mock_get_theme,
        mock_get_pref,
        mock_context,
        mock_pref_manager,
    ):
        """Test the /theme preview command."""
        mock_get_pref.return_value = mock_pref_manager
        mock_get_theme.return_value = "default"
        mock_confirm.return_value = False  # Don't keep the preview

        await handle_theme_command(mock_context, ["preview", "monokai"])

        # Should set theme for preview
        mock_set_theme.assert_any_call("monokai")
        # Should revert since we said no
        mock_set_theme.assert_any_call("default")

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    @patch("mcp_cli.chat.commands.theme.output")
    async def test_preview_command_invalid_theme(
        self, mock_output, mock_get_pref, mock_context, mock_pref_manager
    ):
        """Test preview command with invalid theme."""
        mock_get_pref.return_value = mock_pref_manager

        await handle_theme_command(mock_context, ["preview", "not_a_theme"])

        # Should show error
        mock_output.error.assert_called()
        assert "Invalid theme" in str(mock_output.error.call_args)

    # ===== Edge Cases =====

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    @patch("mcp_cli.chat.commands.theme.output")
    async def test_empty_string_argument(
        self, mock_output, mock_get_pref, mock_context, mock_pref_manager
    ):
        """Test handling of empty string argument."""
        mock_get_pref.return_value = mock_pref_manager

        await handle_theme_command(mock_context, [""])

        # Should show error for invalid theme
        mock_output.error.assert_called()

    @pytest.mark.asyncio
    @patch("chuk_term.ui.prompts.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    async def test_interactive_empty_response(
        self, mock_output, mock_ask, mock_context, mock_pref_manager
    ):
        """Test handling of empty response in interactive mode."""
        mock_pref_manager.get_theme.return_value = "dark"
        mock_ask.return_value = ""  # Empty response should use default

        await interactive_theme_selection(mock_context, mock_pref_manager)

        # Empty response keeps current theme
        mock_output.info.assert_called()
        assert "Keeping current theme" in str(mock_output.info.call_args)
