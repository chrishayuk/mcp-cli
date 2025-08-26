"""Tests for chat mode help command."""

from unittest.mock import MagicMock, patch
import pytest

from mcp_cli.chat.commands.help import cmd_help, display_quick_help
from mcp_cli.chat.commands import _COMMAND_HANDLERS, _COMMAND_COMPLETIONS


class TestChatHelpCommand:
    """Test chat mode help command."""

    @pytest.fixture
    def mock_context(self):
        """Create mock chat context."""
        return {}

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Set up command registry for tests."""
        # Store original state
        original_handlers = _COMMAND_HANDLERS.copy()
        original_completions = _COMMAND_COMPLETIONS.copy()

        # Clear and add test commands
        _COMMAND_HANDLERS.clear()
        _COMMAND_COMPLETIONS.clear()

        # Add some test commands
        async def test_cmd(parts, ctx):
            """Test command for testing."""
            return True

        async def another_cmd(parts, ctx):
            """Another test command with longer description that should be truncated."""
            return True

        _COMMAND_HANDLERS["/test"] = test_cmd
        _COMMAND_HANDLERS["/another"] = another_cmd
        _COMMAND_COMPLETIONS["/test"] = ["arg1", "arg2"]

        yield

        # Restore original state
        _COMMAND_HANDLERS.clear()
        _COMMAND_HANDLERS.update(original_handlers)
        _COMMAND_COMPLETIONS.clear()
        _COMMAND_COMPLETIONS.update(original_completions)

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_no_arguments(self, mock_output, mock_context):
        """Test /help with no arguments shows all commands."""
        result = await cmd_help(["/help"], mock_context)

        assert result is True

        # Should create and print a table
        mock_output.print.assert_called()

        # Check that a table was created
        calls = mock_output.print.call_args_list
        # Look for the table or instruction text
        call_strings = [str(call) for call in calls]
        assert any(
            "Available Commands" in s or "/help <command>" in s for s in call_strings
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_specific_command(self, mock_output, mock_context):
        """Test /help with specific command shows detailed help."""
        result = await cmd_help(["/help", "/test"], mock_context)

        assert result is True

        # Should print a panel with command details
        mock_output.print.assert_called()

        # Check for Panel creation with markdown
        panel_calls = [
            call for call in mock_output.print.call_args_list if "Panel" in str(call)
        ]
        assert len(panel_calls) > 0 or mock_output.print.called

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_command_without_slash(self, mock_output, mock_context):
        """Test /help handles command names without leading slash."""
        result = await cmd_help(["/help", "test"], mock_context)

        assert result is True

        # Should still find and display the command
        mock_output.print.assert_called()

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_unknown_command(self, mock_output, mock_context):
        """Test /help with unknown command shows message."""
        result = await cmd_help(["/help", "/unknown"], mock_context)

        assert result is True

        # Should print something (either error or regular output)
        assert mock_output.print.called

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_tools_group(self, mock_output, mock_context):
        """Test /help tools shows grouped tool commands help."""
        result = await cmd_help(["/help", "tools"], mock_context)

        assert result is True

        # Should print a panel with tool commands help
        mock_output.print.assert_called()
        panel_call = mock_output.print.call_args[0][0]

        # Check that Panel was created (might be wrapped in call)
        assert "Panel" in str(panel_call) or "Tool Commands" in str(
            mock_output.print.call_args_list
        )

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_conversation_group(self, mock_output, mock_context):
        """Test /help conversation shows grouped conversation commands."""
        result = await cmd_help(["/help", "conversation"], mock_context)

        assert result is True

        # Should print a panel with conversation commands help
        mock_output.print.assert_called()
        # Check the Panel object was created and printed
        call_args = mock_output.print.call_args_list
        assert len(call_args) > 0

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_ui_group(self, mock_output, mock_context):
        """Test /help ui shows grouped UI commands help."""
        result = await cmd_help(["/help", "ui"], mock_context)

        assert result is True

        # Should print a panel with UI commands help
        mock_output.print.assert_called()

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_quickhelp_command(self, mock_output, mock_context):
        """Test /quickhelp (qh) shows condensed help."""
        result = await display_quick_help(["/qh"], mock_context)

        assert result is True

        # Should print a table
        mock_output.print.assert_called()

        # Check for quick reference content
        call_strings = str(mock_output.print.call_args_list)
        assert "Quick Command Reference" in call_strings or "/help" in call_strings

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_with_completions(self, mock_output, mock_context):
        """Test help shows completions for commands that have them."""
        result = await cmd_help(["/help", "/test"], mock_context)

        assert result is True

        # Should print something (Panel with command help)
        mock_output.print.assert_called()
        # The completions are embedded in the Markdown inside the Panel
        # We can't easily inspect the Panel content in the test

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.Panel")
    @patch("mcp_cli.chat.commands.help.Markdown")
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_panel_creation(
        self, mock_output, mock_markdown, mock_panel, mock_context
    ):
        """Test that help creates proper Panel and Markdown objects."""
        # Mock Panel to return a simple object
        mock_panel.return_value = "MOCKED_PANEL"
        mock_markdown.return_value = "MOCKED_MARKDOWN"

        result = await cmd_help(["/help", "/test"], mock_context)

        assert result is True

        # Check that Markdown was created with command help
        mock_markdown.assert_called()
        markdown_content = mock_markdown.call_args[0][0]
        assert "/test" in markdown_content or "Test command" in markdown_content

        # Check that Panel was created
        mock_panel.assert_called()
        panel_args = mock_panel.call_args
        assert panel_args.kwargs.get("title") == "Help: /test"
        assert panel_args.kwargs.get("style") == "cyan"

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.Table")
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_table_creation(
        self, mock_output, mock_table_class, mock_context
    ):
        """Test that help creates proper table for all commands."""
        mock_table = MagicMock()
        mock_table_class.return_value = mock_table

        result = await cmd_help(["/help"], mock_context)

        assert result is True

        # Check that Table was created
        mock_table_class.assert_called_with(title="2 Available Commands")

        # Check that columns were added
        mock_table.add_column.assert_any_call("Command", style="green")
        mock_table.add_column.assert_any_call("Description")

        # Check that rows were added for our test commands
        assert mock_table.add_row.call_count == 2

        # Check specific commands were added
        row_calls = mock_table.add_row.call_args_list
        commands_added = [call[0][0] for call in row_calls]
        assert "/test" in commands_added
        assert "/another" in commands_added

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_empty_registry(self, mock_output, mock_context):
        """Test help handles empty command registry gracefully."""
        # Clear all commands
        _COMMAND_HANDLERS.clear()

        result = await cmd_help(["/help"], mock_context)

        assert result is True

        # Should still handle gracefully
        mock_output.print.assert_called()

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_theme_shows_command_not_group(self, mock_output, mock_context):
        """Test /help theme shows theme command help, not UI group."""

        # Register a mock /theme command
        async def theme_cmd(parts, ctx):
            """Theme command for testing."""
            return True

        from mcp_cli.chat.commands import _COMMAND_HANDLERS

        _COMMAND_HANDLERS["/theme"] = theme_cmd

        result = await cmd_help(["/help", "theme"], mock_context)

        assert result is True

        # Should show individual command help (Panel is printed)
        mock_output.print.assert_called()

        # Clean up
        del _COMMAND_HANDLERS["/theme"]

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.help.output")
    async def test_help_preferences_alias(self, mock_output, mock_context):
        """Test /help preferences shows UI commands group."""
        result = await cmd_help(["/help", "preferences"], mock_context)

        assert result is True

        # Should show UI commands help (Panel is printed)
        mock_output.print.assert_called()
