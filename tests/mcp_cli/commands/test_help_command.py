# commands/test_help_command.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from rich.console import Console

from mcp_cli.commands.help import help_action
from mcp_cli.interactive.registry import InteractiveCommandRegistry
from mcp_cli.interactive.commands.help import HelpCommand


class DummyCmd:
    def __init__(self, name, help_text, aliases=None):
        self.name = name
        self.help = help_text
        self.help_text = help_text  # Support both attributes
        self.aliases = aliases or []


@pytest.fixture(autouse=True)
def clear_registry():
    # Force type to dict in case any other test or code polluted it
    InteractiveCommandRegistry._commands = {}
    InteractiveCommandRegistry._aliases = {}
    yield
    InteractiveCommandRegistry._commands = {}
    InteractiveCommandRegistry._aliases = {}


def test_help_action_list_all():
    """Test help_action displays all commands when no specific command given."""
    # Register two dummy commands
    cmd_a = DummyCmd("a", "help A", aliases=["x"])
    cmd_b = DummyCmd("b", "help B", aliases=[])
    InteractiveCommandRegistry.register(cmd_a)
    InteractiveCommandRegistry.register(cmd_b)

    # Mock the output functions
    with patch('mcp_cli.commands.help.output') as mock_output:
        with patch('mcp_cli.commands.help.format_table') as mock_format_table:
            # Mock format_table to return a mock table
            mock_table = Mock()
            mock_format_table.return_value = mock_table
            
            # Call help_action without arguments
            help_action()
            
            # Should have called format_table with the commands data
            mock_format_table.assert_called_once()
            call_args = mock_format_table.call_args
            
            # Check the table data
            table_data = call_args[0][0]  # First positional argument
            assert len(table_data) == 2  # Two commands
            
            # Check first command
            assert table_data[0]["Command"] == "a"
            assert table_data[0]["Aliases"] == "x"
            assert "help A" in table_data[0]["Description"]
            
            # Check second command
            assert table_data[1]["Command"] == "b"
            assert table_data[1]["Aliases"] == "-"
            assert "help B" in table_data[1]["Description"]
            
            # Should print the table
            mock_output.print_table.assert_called_once_with(mock_table)
            
            # Should show hint
            mock_output.hint.assert_called_once()
            hint_text = mock_output.hint.call_args[0][0]
            assert "help <command>" in hint_text


def test_help_action_specific():
    """Test help_action displays specific command help."""
    # Register one dummy command
    cmd = DummyCmd("foo", "Foo does X", aliases=["f"])
    InteractiveCommandRegistry.register(cmd)
    
    # Mock the output module to capture what gets passed to panel
    with patch('mcp_cli.commands.help.output') as mock_output:
        # Call help_action for specific command (with optional console arg)
        help_action("foo")
        
        # Should have called panel with markdown content
        mock_output.panel.assert_called_once()
        
        # Get the arguments passed to panel
        call_args = mock_output.panel.call_args
        content = call_args[0][0]  # First positional argument
        
        # Check the content contains expected text
        assert "## foo" in content
        assert "Foo does X" in content
        
        # Check panel was called with correct title and style
        assert call_args[1]["title"] == "Command Help"
        assert call_args[1]["style"] == "cyan"
        
        # Should also print aliases line
        mock_output.print.assert_called_once()
        print_call = mock_output.print.call_args[0][0]
        assert "Aliases: f" in print_call


def test_help_action_with_console():
    """Test help_action accepts console argument for backward compatibility."""
    # Register one dummy command
    cmd = DummyCmd("test", "Test command", aliases=["t"])
    InteractiveCommandRegistry.register(cmd)
    
    # Mock the output module
    with patch('mcp_cli.commands.help.output') as mock_output:
        # Create a mock console
        mock_console = Mock(spec=Console)
        
        # Call with console argument (should be accepted but ignored)
        help_action("test", console=mock_console)
        
        # Should still use output module, not the console
        mock_output.panel.assert_called_once()


def test_help_action_unknown_command():
    """Test help_action with unknown command shows error."""
    with patch('mcp_cli.commands.help.output') as mock_output:
        help_action("nonexistent")
        
        # Should show error
        mock_output.error.assert_called_once_with("Unknown command: nonexistent")


def test_help_action_no_commands():
    """Test help_action when no commands are registered."""
    # Clear registry (already done by fixture)
    
    with patch('mcp_cli.commands.help.output') as mock_output:
        help_action()
        
        # Should show warning
        mock_output.warning.assert_called_once_with("No commands available")


@pytest.mark.asyncio
async def test_interactive_wrapper():
    """Test HelpCommand wrapper for interactive mode."""
    # Register a command
    cmd_dummy = DummyCmd("foo", "help foo", aliases=[])
    InteractiveCommandRegistry.register(cmd_dummy)
    
    # Mock get_console to return a mock console
    with patch('mcp_cli.interactive.commands.help.get_console') as mock_get_console:
        mock_console = Mock(spec=Console)
        mock_get_console.return_value = mock_console
        
        # Mock help_action to verify it's called correctly
        with patch('mcp_cli.interactive.commands.help.help_action') as mock_help_action:
            help_cmd = HelpCommand()
            
            # Test with no arguments - should show all commands
            await help_cmd.execute([], tool_manager=None)
            mock_help_action.assert_called_with(None, console=mock_console)
            
            # Reset mock
            mock_help_action.reset_mock()
            
            # Test with specific command
            await help_cmd.execute(["foo"], tool_manager=None)
            mock_help_action.assert_called_with("foo", console=mock_console)
            
            # Test with command starting with "/"
            mock_help_action.reset_mock()
            await help_cmd.execute(["/foo"], tool_manager=None)
            # Should strip the leading "/"
            mock_help_action.assert_called_with("foo", console=mock_console)


def test_extract_description():
    """Test description extraction from help text."""
    from mcp_cli.commands.help import _extract_description
    
    # Test with simple description
    assert _extract_description("Simple description") == "Simple description"
    
    # Test with multi-line description
    assert _extract_description("First line\nSecond line") == "First line"
    
    # Test with usage line (should skip)
    assert _extract_description("Usage: foo\nActual description") == "Actual description"
    
    # Test with empty/None
    assert _extract_description(None) == "No description"
    assert _extract_description("") == "No description"
    
    # Test with only whitespace
    assert _extract_description("  \n  \n  ") == "No description"


def test_command_with_no_aliases():
    """Test command display when command has no aliases."""
    cmd = DummyCmd("test", "Test command", aliases=[])
    InteractiveCommandRegistry.register(cmd)
    
    with patch('mcp_cli.commands.help.output') as mock_output:
        with patch('mcp_cli.commands.help.format_table') as mock_format_table:
            mock_format_table.return_value = Mock()
            
            help_action()
            
            # Check that aliases show as "-" when empty
            call_args = mock_format_table.call_args
            table_data = call_args[0][0]
            assert table_data[0]["Aliases"] == "-"


def test_command_with_multiple_aliases():
    """Test command display with multiple aliases."""
    cmd = DummyCmd("test", "Test command", aliases=["t", "tst", "test2"])
    InteractiveCommandRegistry.register(cmd)
    
    with patch('mcp_cli.commands.help.output') as mock_output:
        with patch('mcp_cli.commands.help.format_table') as mock_format_table:
            mock_format_table.return_value = Mock()
            
            help_action()
            
            # Check that multiple aliases are comma-separated
            call_args = mock_format_table.call_args
            table_data = call_args[0][0]
            assert table_data[0]["Aliases"] == "t, tst, test2"