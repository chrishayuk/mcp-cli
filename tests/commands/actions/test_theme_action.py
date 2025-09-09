"""Tests for the theme action command."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch

from mcp_cli.commands.actions.theme import (
    theme_command,
    _interactive_theme_selection,
    _show_theme_preview,
    theme_action_async,
)
from mcp_cli.utils.preferences import Theme


@pytest.fixture
def mock_pref_manager():
    """Create a mock preference manager."""
    manager = MagicMock()
    manager.get_theme.return_value = "default"
    manager.set_theme.return_value = None
    return manager


def test_theme_command_list_themes(mock_pref_manager):
    """Test listing all available themes."""
    with patch(
        "mcp_cli.commands.actions.theme.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch("mcp_cli.commands.actions.theme.output") as mock_output:
            theme_command(list_themes=True)

            mock_output.rule.assert_called_with("Available Themes")
            # Should print all themes
            assert mock_output.print.call_count >= len(Theme)
            mock_output.hint.assert_called_with("Use '/theme <name>' to switch themes")


def test_theme_command_no_args_lists_themes(mock_pref_manager):
    """Test that no arguments also lists themes."""
    with patch(
        "mcp_cli.commands.actions.theme.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch("mcp_cli.commands.actions.theme.output") as mock_output:
            theme_command()

            mock_output.rule.assert_called_with("Available Themes")
            mock_output.hint.assert_called()


def test_theme_command_list_themes_shows_current(mock_pref_manager):
    """Test that listing themes highlights the current theme."""
    mock_pref_manager.get_theme.return_value = "dark"

    with patch(
        "mcp_cli.commands.actions.theme.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch("mcp_cli.commands.actions.theme.output") as mock_output:
            theme_command(list_themes=True)

            # Check that current theme is marked
            info_calls = [str(call) for call in mock_output.info.call_args_list]
            assert any("dark (current)" in str(call) for call in info_calls)


def test_theme_command_switch_valid_theme(mock_pref_manager):
    """Test switching to a valid theme."""
    with patch(
        "mcp_cli.commands.actions.theme.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch("mcp_cli.commands.actions.theme.set_theme") as mock_set_theme:
            with patch("mcp_cli.commands.actions.theme.output") as mock_output:
                theme_command(theme_name="dark")

                mock_set_theme.assert_called_with("dark")
                mock_pref_manager.set_theme.assert_called_with("dark")
                mock_output.success.assert_called_with("Theme switched to: dark")


def test_theme_command_switch_invalid_theme(mock_pref_manager):
    """Test switching to an invalid theme."""
    with patch(
        "mcp_cli.commands.actions.theme.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch("mcp_cli.commands.actions.theme.output") as mock_output:
            theme_command(theme_name="invalid-theme")

            mock_output.error.assert_called_with("Invalid theme: invalid-theme")
            mock_output.hint.assert_called()
            mock_pref_manager.set_theme.assert_not_called()


def test_theme_command_switch_case_insensitive(mock_pref_manager):
    """Test that theme switching is case insensitive."""
    with patch(
        "mcp_cli.commands.actions.theme.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch("mcp_cli.commands.actions.theme.set_theme") as mock_set_theme:
            with patch("mcp_cli.commands.actions.theme.output") as mock_output:
                theme_command(theme_name="DARK")

                mock_set_theme.assert_called_with("dark")
                mock_pref_manager.set_theme.assert_called_with("dark")
                mock_output.success.assert_called()


def test_theme_command_switch_error_handling(mock_pref_manager):
    """Test error handling when theme switch fails."""
    with patch(
        "mcp_cli.commands.actions.theme.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch(
            "mcp_cli.commands.actions.theme.set_theme",
            side_effect=Exception("Test error"),
        ):
            with patch("mcp_cli.commands.actions.theme.output") as mock_output:
                theme_command(theme_name="dark")

                mock_output.error.assert_called_with(
                    "Failed to switch theme: Test error"
                )


def test_theme_command_select_interactive(mock_pref_manager):
    """Test interactive theme selection."""
    with patch(
        "mcp_cli.commands.actions.theme.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch("asyncio.run") as mock_run:
            theme_command(select=True)

            mock_run.assert_called_once()
            # Check that _interactive_theme_selection was passed
            call_args = mock_run.call_args[0][0]
            assert hasattr(call_args, "__name__") or asyncio.iscoroutine(call_args)


@pytest.mark.asyncio
async def test_interactive_theme_selection_numeric_input(mock_pref_manager):
    """Test interactive selection with numeric input."""
    with patch("mcp_cli.commands.actions.theme.output") as mock_output:
        with patch("mcp_cli.commands.actions.theme.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"
            with patch("mcp_cli.commands.actions.theme.ask", return_value="2"):
                with patch(
                    "mcp_cli.commands.actions.theme.set_theme"
                ) as mock_set_theme:
                    await _interactive_theme_selection(mock_pref_manager)

                    # Should select the second theme (dark)
                    mock_set_theme.assert_called_with("dark")
                    mock_pref_manager.set_theme.assert_called_with("dark")
                    mock_output.success.assert_called()


@pytest.mark.asyncio
async def test_interactive_theme_selection_name_input(mock_pref_manager):
    """Test interactive selection with theme name input."""
    with patch("mcp_cli.commands.actions.theme.output"):
        with patch("mcp_cli.commands.actions.theme.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"
            with patch("mcp_cli.commands.actions.theme.ask", return_value="monokai"):
                with patch(
                    "mcp_cli.commands.actions.theme.set_theme"
                ) as mock_set_theme:
                    await _interactive_theme_selection(mock_pref_manager)

                    mock_set_theme.assert_called_with("monokai")
                    mock_pref_manager.set_theme.assert_called_with("monokai")


@pytest.mark.asyncio
async def test_interactive_theme_selection_invalid_numeric(mock_pref_manager):
    """Test interactive selection with invalid numeric input."""
    with patch("mcp_cli.commands.actions.theme.output") as mock_output:
        with patch("mcp_cli.commands.actions.theme.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"
            with patch("mcp_cli.commands.actions.theme.ask", return_value="99"):
                await _interactive_theme_selection(mock_pref_manager)

                mock_output.error.assert_called()
                mock_pref_manager.set_theme.assert_not_called()


@pytest.mark.asyncio
async def test_interactive_theme_selection_invalid_name(mock_pref_manager):
    """Test interactive selection with invalid theme name."""
    with patch("mcp_cli.commands.actions.theme.output") as mock_output:
        with patch("mcp_cli.commands.actions.theme.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"
            with patch(
                "mcp_cli.commands.actions.theme.ask", return_value="nonexistent"
            ):
                await _interactive_theme_selection(mock_pref_manager)

                mock_output.error.assert_called_with("Unknown theme: nonexistent")
                mock_output.hint.assert_called()
                mock_pref_manager.set_theme.assert_not_called()


@pytest.mark.asyncio
async def test_interactive_theme_selection_same_theme(mock_pref_manager):
    """Test selecting the same theme that's already active."""
    mock_pref_manager.get_theme.return_value = "default"

    with patch("mcp_cli.commands.actions.theme.output") as mock_output:
        with patch("mcp_cli.commands.actions.theme.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"
            with patch("mcp_cli.commands.actions.theme.ask", return_value="default"):
                await _interactive_theme_selection(mock_pref_manager)

                mock_output.info.assert_called_with("Already using theme: default")
                mock_pref_manager.set_theme.assert_not_called()


@pytest.mark.asyncio
async def test_interactive_theme_selection_shows_preview(mock_pref_manager):
    """Test that theme preview is shown after selection."""
    with patch("mcp_cli.commands.actions.theme.output"):
        with patch("mcp_cli.commands.actions.theme.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"
            with patch("mcp_cli.commands.actions.theme.ask", return_value="dark"):
                with patch("mcp_cli.commands.actions.theme.set_theme"):
                    with patch(
                        "mcp_cli.commands.actions.theme._show_theme_preview"
                    ) as mock_preview:
                        await _interactive_theme_selection(mock_pref_manager)

                        mock_preview.assert_called_once()


def test_show_theme_preview():
    """Test theme preview display."""
    with patch("mcp_cli.commands.actions.theme.output") as mock_output:
        _show_theme_preview()

        mock_output.print.assert_called_with("Theme Preview:")
        mock_output.info.assert_called_with("Information message")
        mock_output.success.assert_called_with("Success message")
        mock_output.warning.assert_called_with("Warning message")
        mock_output.error.assert_called_with("Error message")
        mock_output.hint.assert_called_with("Hint message")


@pytest.mark.asyncio
async def test_theme_action_async_no_args():
    """Test async theme action with no arguments."""
    with patch("mcp_cli.commands.actions.theme.theme_command") as mock_command:
        await theme_action_async([])

        mock_command.assert_called_with(list_themes=True)


@pytest.mark.asyncio
async def test_theme_action_async_with_theme_name():
    """Test async theme action with theme name."""
    with patch("mcp_cli.commands.actions.theme.theme_command") as mock_command:
        await theme_action_async(["dark"])

        mock_command.assert_called_with(theme_name="dark")


@pytest.mark.asyncio
async def test_theme_action_async_multiple_args():
    """Test async theme action with multiple arguments (uses first one)."""
    with patch("mcp_cli.commands.actions.theme.theme_command") as mock_command:
        await theme_action_async(["dark", "extra", "args"])

        mock_command.assert_called_with(theme_name="dark")
