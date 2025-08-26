"""Tests for the main help_action function used across all modes."""

import pytest
from unittest.mock import MagicMock, patch

from mcp_cli.commands.help import help_action, _extract_description
from mcp_cli.interactive.registry import InteractiveCommandRegistry


class DummyTestCommand:
    """Dummy command class for testing."""

    def __init__(self, name, help_text="", aliases=None):
        self.name = name
        self.help = help_text
        self.aliases = aliases or []


class TestHelpAction:
    """Test the help_action function."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Set up test registry."""
        # Store original
        original_commands = InteractiveCommandRegistry._commands.copy()
        original_aliases = InteractiveCommandRegistry._aliases.copy()

        # Clear and setup test commands
        InteractiveCommandRegistry._commands.clear()
        InteractiveCommandRegistry._aliases.clear()

        # Force help module to use InteractiveCommandRegistry
        import mcp_cli.commands.help

        mcp_cli.commands.help.Registry = InteractiveCommandRegistry

        # Add test commands
        cmd1 = DummyTestCommand("test", "Test command help text", ["t", "tst"])
        cmd2 = DummyTestCommand(
            "another",
            "Another command with a much longer help text that spans multiple lines\nwith additional details",
        )
        cmd3 = DummyTestCommand("nohelp", "")  # Command with empty help

        InteractiveCommandRegistry._commands = {
            "test": cmd1,
            "another": cmd2,
            "nohelp": cmd3,
        }

        yield

        # Restore
        InteractiveCommandRegistry._commands = original_commands
        InteractiveCommandRegistry._aliases = original_aliases

    @patch("mcp_cli.commands.help.output")
    @patch("mcp_cli.commands.help.format_table")
    def test_help_action_show_all(self, mock_format_table, mock_output):
        """Test help_action with no command shows all commands."""
        mock_table = MagicMock()
        mock_format_table.return_value = mock_table

        help_action(None)

        # Should create a table
        mock_format_table.assert_called_once()
        table_data = mock_format_table.call_args[0][0]

        # Should have all test commands
        assert len(table_data) == 3
        commands = {row["Command"] for row in table_data}
        assert commands == {"test", "another", "nohelp"}

        # Check aliases are included
        test_row = next(r for r in table_data if r["Command"] == "test")
        assert test_row["Aliases"] == "t, tst"

        # Check descriptions
        assert "Test command help text" in test_row["Description"]

        # Should print the table
        mock_output.print_table.assert_called_once_with(mock_table)

    @patch("mcp_cli.commands.help.output")
    def test_help_action_specific_command(self, mock_output):
        """Test help_action with specific command name."""
        help_action("test")

        # Should show panel with command details
        mock_output.panel.assert_called_once()
        panel_content = mock_output.panel.call_args[0][0]

        assert "test" in panel_content
        assert "Test command help text" in panel_content

        # Should show aliases
        mock_output.print.assert_called()
        print_content = str(mock_output.print.call_args_list)
        assert "t, tst" in print_content or "Aliases" in print_content

    @patch("mcp_cli.commands.help.output")
    def test_help_action_unknown_command(self, mock_output):
        """Test help_action with unknown command."""
        help_action("nonexistent")

        # Should show error
        mock_output.error.assert_called_once()
        error_msg = mock_output.error.call_args[0][0]
        assert "Unknown command" in error_msg
        assert "nonexistent" in error_msg

    @patch("mcp_cli.commands.help.output")
    def test_help_action_empty_registry(self, mock_output):
        """Test help_action with empty command registry."""
        # Clear commands
        InteractiveCommandRegistry._commands.clear()

        help_action(None)

        # Should show warning
        mock_output.warning.assert_called_once_with("No commands available")

    @patch("mcp_cli.commands.help.output")
    def test_help_action_command_no_aliases(self, mock_output):
        """Test help_action for command without aliases."""
        help_action("another")

        # Should show panel
        mock_output.panel.assert_called_once()

        # Should not show aliases line (since it has none)
        if mock_output.print.called:
            # If aliases are shown, they should be empty
            for call in mock_output.print.call_args_list:
                call_str = str(call)
                if "Aliases" in call_str:
                    assert False, "Should not show aliases for command without aliases"

    @patch("mcp_cli.commands.help.output")
    def test_help_action_command_empty_help(self, mock_output):
        """Test help_action for command with empty help text."""
        help_action("nohelp")

        # Should still show panel
        mock_output.panel.assert_called_once()
        panel_content = mock_output.panel.call_args[0][0]

        # Should have command name
        assert "nohelp" in panel_content
        # May have default text
        assert "No description" in panel_content or panel_content

    @patch("mcp_cli.commands.help.output")
    def test_help_action_with_console_arg(self, mock_output):
        """Test help_action accepts optional console argument for compatibility."""
        mock_console = MagicMock()

        # Should accept console argument but not use it
        help_action("test", console=mock_console)

        # Should still work normally
        mock_output.panel.assert_called_once()

    def test_extract_description(self):
        """Test _extract_description helper function."""
        # Test simple single line
        assert _extract_description("Simple help text") == "Simple help text"

        # Test multi-line - should take first line
        multi = "First line\nSecond line\nThird line"
        assert _extract_description(multi) == "First line"

        # Test with usage line
        with_usage = "Usage: command [options]\nActual help text\nMore details"
        assert _extract_description(with_usage) == "Actual help text"

        # Test empty - returns default
        assert _extract_description("") == "No description"

        # Test only whitespace - returns default
        assert _extract_description("   \n  \n  ") == "No description"

        # Test None - returns default
        assert _extract_description(None) == "No description"

    @patch("mcp_cli.commands.help.output")
    @patch("mcp_cli.commands.help.format_table")
    def test_help_action_table_formatting(self, mock_format_table, mock_output):
        """Test that help_action formats the table correctly."""
        mock_format_table.return_value = MagicMock()

        help_action(None)

        # Check format_table was called with correct parameters
        mock_format_table.assert_called_once()
        call_args = mock_format_table.call_args

        # Check table data structure
        table_data = call_args[0][0]
        assert all("Command" in row for row in table_data)
        assert all("Aliases" in row for row in table_data)
        assert all("Description" in row for row in table_data)

        # Check kwargs
        assert call_args.kwargs["title"] == "Available Commands"
        assert call_args.kwargs["columns"] == ["Command", "Aliases", "Description"]

    @patch("mcp_cli.commands.help.output")
    def test_help_action_panel_formatting(self, mock_output):
        """Test that help_action formats the panel correctly."""
        help_action("test")

        # Check panel was called with correct parameters
        mock_output.panel.assert_called_once()
        call_args = mock_output.panel.call_args

        # Check content includes markdown formatting
        content = call_args[0][0]
        assert "## test" in content
        assert "Test command help text" in content

        # Check panel kwargs
        assert call_args.kwargs["title"] == "Command Help"
        assert call_args.kwargs["style"] == "cyan"

    def test_extract_description_edge_cases(self):
        """Test _extract_description with edge cases."""
        # Test with only "Usage:" line - returns default
        assert _extract_description("Usage: foo") == "No description"

        # Test with leading/trailing whitespace
        assert _extract_description("  Trimmed text  ") == "Trimmed text"

        # Test with multiple usage lines
        text = "Usage: foo\nUsage: bar\nActual description"
        assert _extract_description(text) == "Actual description"

        # Test with blank lines
        text = "\n\n\nActual text\n\n"
        assert _extract_description(text) == "Actual text"
