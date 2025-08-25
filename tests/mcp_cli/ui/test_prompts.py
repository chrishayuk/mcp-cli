"""Unit tests for user prompt and interaction utilities."""

import sys
import json
import termios
import tty
from unittest.mock import Mock, patch, MagicMock, call
import pytest
from typing import List, Dict, Any

from mcp_cli.ui.prompts import (
    ask,
    confirm,
    ask_number,
    select_from_list,
    select_multiple,
    prompt_for_tool_confirmation,
    prompt_for_retry,
    create_menu,
)


# PromptStyle tests removed as it's no longer exposed
class TestPromptStyle:
    """Test PromptStyle constants - SKIP as no longer exposed."""
    
    @pytest.mark.skip(reason="PromptStyle no longer exposed after chuk-term migration")
    def test_prompt_styles(self):
        """Test that all prompt styles are defined."""
        # PromptStyle constants removed after chuk-term migration
        pass


@pytest.mark.skip(reason="_get_key no longer exposed after chuk-term migration")
class TestGetKey:
    """Test _get_key function for keyboard input."""
    
    @patch('sys.platform', 'win32')
    def test_get_key_windows_normal(self):
        """Test getting normal key on Windows."""
        # Mock msvcrt inside sys.modules
        mock_msvcrt = Mock()
        mock_msvcrt.getch.return_value = b'a'
        
        with patch.dict('sys.modules', {'msvcrt': mock_msvcrt}):
            assert _get_key() == 'a'
            mock_msvcrt.getch.assert_called_once()
    
    @patch('sys.platform', 'win32')
    def test_get_key_windows_enter(self):
        """Test getting Enter key on Windows."""
        mock_msvcrt = Mock()
        mock_msvcrt.getch.return_value = b'\r'
        
        with patch.dict('sys.modules', {'msvcrt': mock_msvcrt}):
            assert _get_key() == 'enter'
    
    @patch('sys.platform', 'win32')
    def test_get_key_windows_space(self):
        """Test getting Space key on Windows."""
        mock_msvcrt = Mock()
        mock_msvcrt.getch.return_value = b' '
        
        with patch.dict('sys.modules', {'msvcrt': mock_msvcrt}):
            assert _get_key() == 'space'
    
    @patch('sys.platform', 'win32')
    def test_get_key_windows_special(self):
        """Test getting arrow keys on Windows."""
        mock_msvcrt = Mock()
        mock_msvcrt.getch.side_effect = [b'\xe0', b'H']  # Up arrow
        
        with patch.dict('sys.modules', {'msvcrt': mock_msvcrt}):
            assert _get_key() == 'up'
        
        mock_msvcrt.getch.side_effect = [b'\xe0', b'P']  # Down arrow
        with patch.dict('sys.modules', {'msvcrt': mock_msvcrt}):
            assert _get_key() == 'down'
    
    @patch('sys.platform', 'win32')
    def test_get_key_windows_ctrl_c(self):
        """Test Ctrl+C on Windows."""
        mock_msvcrt = Mock()
        mock_msvcrt.getch.return_value = b'\x03'
        
        with patch.dict('sys.modules', {'msvcrt': mock_msvcrt}):
            with pytest.raises(KeyboardInterrupt):
                _get_key()
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdin')
    @patch('termios.tcsetattr')
    @patch('termios.tcgetattr')
    @patch('tty.setraw')
    def test_get_key_unix_normal(self, mock_setraw, mock_tcgetattr, mock_tcsetattr, mock_stdin):
        """Test getting normal key on Unix."""
        mock_stdin.fileno.return_value = 0
        mock_stdin.read.return_value = 'a'
        mock_tcgetattr.return_value = []
        
        assert _get_key() == 'a'
        mock_setraw.assert_called_once()
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdin')
    @patch('termios.tcsetattr')
    @patch('termios.tcgetattr')
    @patch('tty.setraw')
    def test_get_key_unix_arrow(self, mock_setraw, mock_tcgetattr, mock_tcsetattr, mock_stdin):
        """Test getting arrow keys on Unix."""
        mock_stdin.fileno.return_value = 0
        mock_stdin.read.side_effect = ['\x1b', '[', 'A']  # Up arrow
        mock_tcgetattr.return_value = []
        
        assert _get_key() == 'up'
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdin')
    @patch('termios.tcsetattr')
    @patch('termios.tcgetattr')
    @patch('tty.setraw')
    def test_get_key_unix_enter(self, mock_setraw, mock_tcgetattr, mock_tcsetattr, mock_stdin):
        """Test getting Enter key on Unix."""
        mock_stdin.fileno.return_value = 0
        mock_stdin.read.return_value = '\n'
        mock_tcgetattr.return_value = []
        
        assert _get_key() == 'enter'
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdin')
    @patch('termios.tcsetattr')
    @patch('termios.tcgetattr')
    @patch('tty.setraw')
    def test_get_key_unix_ctrl_c(self, mock_setraw, mock_tcgetattr, mock_tcsetattr, mock_stdin):
        """Test Ctrl+C on Unix."""
        mock_stdin.fileno.return_value = 0
        mock_stdin.read.return_value = '\x03'
        mock_tcgetattr.return_value = []
        
        with pytest.raises(KeyboardInterrupt):
            _get_key()
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdin')
    @patch('termios.tcsetattr')
    @patch('termios.tcgetattr')
    @patch('tty.setraw')
    def test_get_key_unix_ctrl_d(self, mock_setraw, mock_tcgetattr, mock_tcsetattr, mock_stdin):
        """Test Ctrl+D on Unix."""
        mock_stdin.fileno.return_value = 0
        mock_stdin.read.return_value = '\x04'
        mock_tcgetattr.return_value = []
        
        with pytest.raises(EOFError):
            _get_key()


class TestAsk:
    """Test ask function for text input."""
    
    @patch('mcp_cli.ui.prompts.get_theme')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.Prompt')
    def test_ask_basic(self, mock_prompt_class, mock_ui, mock_get_theme):
        """Test basic text input."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_prompt = Mock()
        mock_prompt_class.ask.return_value = "user input"
        mock_console = Mock()
        mock_ui.get_raw_console.return_value = mock_console
        
        result = ask("Enter text:")
        
        assert result == "user input"
        mock_prompt_class.ask.assert_called_once()
        assert "[bold cyan]Enter text:[/]" in str(mock_prompt_class.ask.call_args)
    
    @patch('mcp_cli.ui.prompts.get_theme')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.Prompt')
    def test_ask_minimal_theme(self, mock_prompt_class, mock_ui, mock_get_theme):
        """Test text input with minimal theme."""
        mock_theme = Mock()
        mock_theme.name = "minimal"
        mock_get_theme.return_value = mock_theme
        
        mock_prompt_class.ask.return_value = "user input"
        mock_console = Mock()
        mock_ui.get_raw_console.return_value = mock_console
        
        result = ask("Enter text:")
        
        assert result == "user input"
        # Minimal theme should not add styling
        call_args = mock_prompt_class.ask.call_args
        assert call_args[0][0] == "Enter text:"  # No styling
    
    @patch('mcp_cli.ui.prompts.get_theme')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.Prompt')
    def test_ask_with_default(self, mock_prompt_class, mock_ui, mock_get_theme):
        """Test text input with default value."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_prompt_class.ask.return_value = None  # User pressed Enter
        mock_console = Mock()
        mock_ui.get_raw_console.return_value = mock_console
        
        result = ask("Enter text:", default="default value")
        
        assert result == "default value"
    
    @patch('mcp_cli.ui.prompts.get_theme')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.Prompt')
    def test_ask_password(self, mock_prompt_class, mock_ui, mock_get_theme):
        """Test password input."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_prompt_class.ask.return_value = "secret"
        mock_console = Mock()
        mock_ui.get_raw_console.return_value = mock_console
        
        result = ask("Enter password:", password=True)
        
        assert result == "secret"
        call_kwargs = mock_prompt_class.ask.call_args[1]
        assert call_kwargs['password'] is True
    
    @patch('mcp_cli.ui.prompts.get_theme')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.Prompt')
    def test_ask_keyboard_interrupt(self, mock_prompt_class, mock_ui, mock_get_theme):
        """Test handling Ctrl+C during input."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_prompt_class.ask.side_effect = KeyboardInterrupt()
        mock_console = Mock()
        mock_ui.get_raw_console.return_value = mock_console
        
        result = ask("Enter text:", default="fallback")
        assert result == "fallback"
        
        # Without default, should return empty string
        result = ask("Enter text:")
        assert result == ""


class TestConfirm:
    """Test confirm function for yes/no prompts."""
    
    @patch('mcp_cli.ui.prompts.get_theme')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.Confirm')
    def test_confirm_yes(self, mock_confirm_class, mock_ui, mock_get_theme):
        """Test confirmation with yes."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_confirm_class.ask.return_value = True
        mock_console = Mock()
        mock_ui.get_raw_console.return_value = mock_console
        
        result = confirm("Continue?")
        
        assert result is True
        mock_confirm_class.ask.assert_called_once()
    
    @patch('mcp_cli.ui.prompts.get_theme')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.Confirm')
    def test_confirm_no(self, mock_confirm_class, mock_ui, mock_get_theme):
        """Test confirmation with no."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_confirm_class.ask.return_value = False
        mock_console = Mock()
        mock_ui.get_raw_console.return_value = mock_console
        
        result = confirm("Continue?")
        
        assert result is False
    
    @patch('mcp_cli.ui.prompts.get_theme')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.Confirm')
    def test_confirm_interrupt(self, mock_confirm_class, mock_ui, mock_get_theme):
        """Test handling interrupt during confirmation."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_confirm_class.ask.side_effect = KeyboardInterrupt()
        mock_console = Mock()
        mock_ui.get_raw_console.return_value = mock_console
        
        result = confirm("Continue?", default=True)
        assert result is True  # Returns default on interrupt


class TestAskNumber:
    """Test ask_number function for numeric input."""
    
    @patch('mcp_cli.ui.prompts.get_theme')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.FloatPrompt')
    def test_ask_number_float(self, mock_float_prompt, mock_ui, mock_get_theme):
        """Test float number input."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_float_prompt.ask.return_value = 3.14
        mock_console = Mock()
        mock_ui.get_raw_console.return_value = mock_console
        
        result = ask_number("Enter value:")
        
        assert result == 3.14
        mock_float_prompt.ask.assert_called_once()
    
    @patch('mcp_cli.ui.prompts.get_theme')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.IntPrompt')
    def test_ask_number_integer(self, mock_int_prompt, mock_ui, mock_get_theme):
        """Test integer number input."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_int_prompt.ask.return_value = 42
        mock_console = Mock()
        mock_ui.get_raw_console.return_value = mock_console
        
        result = ask_number("Enter value:", integer=True)
        
        assert result == 42
        mock_int_prompt.ask.assert_called_once()
    
    @patch('mcp_cli.ui.prompts.get_theme')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.FloatPrompt')
    def test_ask_number_with_range(self, mock_float_prompt, mock_ui, mock_get_theme):
        """Test number input with min/max validation."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_console = Mock()
        mock_ui.get_raw_console.return_value = mock_console
        
        # First return value too small, then valid
        mock_float_prompt.ask.side_effect = [0.5, 5.0]
        
        result = ask_number("Enter value:", min_value=1.0, max_value=10.0)
        
        assert result == 5.0
        assert mock_float_prompt.ask.call_count == 2
        mock_ui.warning.assert_called_once_with("Value must be at least 1.0")


class TestSelectFromList:
    """Test select_from_list function."""
    
    def test_select_empty_list(self):
        """Test selecting from empty list raises error."""
        with pytest.raises(ValueError, match="No choices provided"):
            select_from_list("Choose:", [])
    
    @patch('mcp_cli.ui.prompts.ask')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_select_manual_numeric(self, mock_get_theme, mock_ui, mock_ask):
        """Test manual selection with number."""
        mock_theme = Mock()
        mock_theme.name = "minimal"  # Force manual mode
        mock_get_theme.return_value = mock_theme
        
        mock_ask.return_value = "2"
        
        result = select_from_list("Choose:", ["Option A", "Option B", "Option C"])
        
        assert result == "Option B"
        # Check that the option was displayed (format may vary)
        calls = [str(call) for call in mock_ui.print.call_args_list]
        assert any("Option B" in call and "[2]" in call for call in calls)
    
    @patch('mcp_cli.ui.prompts.ask')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_select_manual_by_name(self, mock_get_theme, mock_ui, mock_ask):
        """Test manual selection by entering choice name."""
        mock_theme = Mock()
        mock_theme.name = "minimal"
        mock_get_theme.return_value = mock_theme
        
        mock_ask.return_value = "Option C"
        
        result = select_from_list("Choose:", ["Option A", "Option B", "Option C"])
        
        assert result == "Option C"
    
    @patch('mcp_cli.ui.prompts.ask')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_select_with_default(self, mock_get_theme, mock_ui, mock_ask):
        """Test selection with default value."""
        mock_theme = Mock()
        mock_theme.name = "minimal"
        mock_get_theme.return_value = mock_theme
        
        mock_ask.return_value = ""  # User presses Enter
        
        result = select_from_list(
            "Choose:", 
            ["Option A", "Option B", "Option C"],
            default="Option B"
        )
        
        assert result == "Option B"
        # Check that default is marked
        mock_ui.print.assert_any_call("  → [2] Option B")
    
    @patch('mcp_cli.ui.prompts.ask')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_select_allow_custom(self, mock_get_theme, mock_ui, mock_ask):
        """Test selection with custom input allowed."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_ask.return_value = "Custom Value"
        
        result = select_from_list(
            "Choose:", 
            ["Option A", "Option B"],
            allow_custom=True
        )
        
        assert result == "Custom Value"


class TestSelectMultiple:
    """Test select_multiple function."""
    
    def test_select_multiple_empty_list(self):
        """Test selecting from empty list raises error."""
        with pytest.raises(ValueError, match="No choices provided"):
            select_multiple("Choose:", [])
    
    @patch('mcp_cli.ui.prompts.ask')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_select_multiple_basic(self, mock_get_theme, mock_ui, mock_ask):
        """Test basic multiple selection."""
        mock_theme = Mock()
        mock_theme.name = "minimal"
        mock_get_theme.return_value = mock_theme
        
        # Select items 1 and 3, then done
        mock_ask.side_effect = ["1 3", ""]
        
        result = select_multiple("Choose:", ["A", "B", "C", "D"])
        
        assert set(result) == {"A", "C"}
    
    @patch('mcp_cli.ui.prompts.ask')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_select_multiple_with_range(self, mock_get_theme, mock_ui, mock_ask):
        """Test multiple selection with range."""
        mock_theme = Mock()
        mock_theme.name = "minimal"
        mock_get_theme.return_value = mock_theme
        
        # Select range 1-3
        mock_ask.side_effect = ["1-3", ""]
        
        result = select_multiple("Choose:", ["A", "B", "C", "D"])
        
        assert set(result) == {"A", "B", "C"}
    
    @patch('mcp_cli.ui.prompts.ask')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_select_multiple_all_none(self, mock_get_theme, mock_ui, mock_ask):
        """Test select all and none commands."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        # Select all, then none, then specific items
        mock_ask.side_effect = ["all", "none", "2", ""]
        
        result = select_multiple("Choose:", ["A", "B", "C"])
        
        assert result == ["B"]  # Only item 2 selected after all/none/2
    
    @patch('mcp_cli.ui.prompts.ask')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_select_multiple_min_max(self, mock_get_theme, mock_ui, mock_ask):
        """Test selection with min/max constraints."""
        mock_theme = Mock()
        mock_theme.name = "minimal"
        mock_get_theme.return_value = mock_theme
        
        # Try to select too few, then select enough
        # Need more ask calls for the selection loop
        mock_ask.side_effect = ["1", "", "2", ""]
        
        result = select_multiple(
            "Choose:", 
            ["A", "B", "C", "D"],
            min_selections=2,
            max_selections=3
        )
        
        assert set(result) == {"A", "B"}
        mock_ui.warning.assert_called_with("Please select at least 2 items")


class TestPromptForToolConfirmation:
    """Test prompt_for_tool_confirmation function."""
    
    @patch('mcp_cli.ui.prompts.confirm')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_tool_confirmation_approved(self, mock_get_theme, mock_ui, mock_confirm):
        """Test tool confirmation approved."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_confirm.return_value = True
        
        result = prompt_for_tool_confirmation(
            "test_tool",
            {"arg1": "value1", "arg2": 42},
            "A test tool"
        )
        
        assert result is True
        mock_ui.print.assert_any_call("Tool: [cyan]test_tool[/cyan]")
        mock_ui.print.assert_any_call("Description: A test tool")
        mock_confirm.assert_called_once()
    
    @patch('mcp_cli.ui.prompts.confirm')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_tool_confirmation_minimal(self, mock_get_theme, mock_ui, mock_confirm):
        """Test tool confirmation with minimal theme."""
        mock_theme = Mock()
        mock_theme.name = "minimal"
        mock_get_theme.return_value = mock_theme
        
        mock_confirm.return_value = False
        
        result = prompt_for_tool_confirmation("test_tool", {})
        
        assert result is False
        mock_ui.print.assert_any_call("\nTool Execution Request")
        mock_ui.print.assert_any_call("Tool: test_tool")


class TestPromptForRetry:
    """Test prompt_for_retry function."""
    
    @patch('mcp_cli.ui.prompts.confirm')
    @patch('mcp_cli.ui.prompts.ui')
    def test_retry_with_attempts_left(self, mock_ui, mock_confirm):
        """Test retry prompt with attempts remaining."""
        mock_confirm.return_value = True
        
        error = Exception("Connection failed")
        result = prompt_for_retry(error, 2, 5)
        
        assert result is True
        mock_ui.error.assert_called_with("Attempt 2/5 failed: Connection failed")
        mock_confirm.assert_called_with(
            "Retry? (3 attempts remaining)",
            default=True,
            style=None  # PromptStyle no longer used
        )
    
    @patch('mcp_cli.ui.prompts.confirm')
    @patch('mcp_cli.ui.prompts.ui')
    def test_retry_max_attempts_reached(self, mock_ui, mock_confirm):
        """Test retry prompt when max attempts reached."""
        error = Exception("Connection failed")
        result = prompt_for_retry(error, 5, 5)
        
        assert result is False
        mock_ui.info.assert_called_with("Maximum attempts reached")
        mock_confirm.assert_not_called()


class TestCreateMenu:
    """Test create_menu function."""
    
    @patch('mcp_cli.ui.prompts.ask')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_create_menu_basic(self, mock_get_theme, mock_ui, mock_ask):
        """Test basic menu creation."""
        mock_theme = Mock()
        mock_theme.name = "minimal"
        mock_get_theme.return_value = mock_theme
        
        mock_ask.return_value = "1"
        
        options = {
            "option1": "First option",
            "option2": "Second option"
        }
        
        result = create_menu("Main Menu", options, back_option=False, quit_option=False)
        
        assert result == "option1"
        mock_ui.print.assert_any_call("\nMain Menu")
        mock_ui.print.assert_any_call("-" * len("Main Menu"))
        mock_ui.print.assert_any_call("[1] option1 - First option")
    
    @patch('mcp_cli.ui.prompts.ask')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_create_menu_with_back_quit(self, mock_get_theme, mock_ui, mock_ask):
        """Test menu with back and quit options."""
        mock_theme = Mock()
        mock_theme.name = "minimal"
        mock_get_theme.return_value = mock_theme
        
        mock_ask.return_value = "3"  # Select quit
        
        options = {"option1": "First option"}
        
        result = create_menu("Menu", options, back_option=True, quit_option=True)
        
        assert result == "quit"
        mock_ui.print.assert_any_call("[2] back - Go back")
        mock_ui.print.assert_any_call("[3] quit - Exit")
    
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_create_menu_rich_table(self, mock_get_theme, mock_ui):
        """Test menu creation with rich table for non-minimal theme."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        # Mock the table
        with patch('mcp_cli.ui.prompts.Table') as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table
            
            with patch('mcp_cli.ui.prompts.ask') as mock_ask:
                mock_ask.return_value = "option1"
                
                options = {"option1": "First option"}
                result = create_menu("Menu", options, back_option=False, quit_option=False)
                
                assert result == "option1"
                mock_table_class.assert_called_once_with(title="Menu", show_header=True)
                mock_table.add_row.assert_called_with("[1] option1", "First option")
                mock_ui.print_table.assert_called_once_with(mock_table)


@pytest.mark.skip(reason="_interactive_select no longer exposed after chuk-term migration")
class TestInteractiveSelect:
    """Test _interactive_select function."""
    
    @patch('sys.stdout')
    @patch('mcp_cli.ui.prompts._get_key')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_interactive_select_arrow_navigation(self, mock_get_theme, mock_ui, mock_get_key, mock_stdout):
        """Test interactive selection with arrow keys."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        # Simulate: down, down, enter
        mock_get_key.side_effect = ['down', 'down', 'enter']
        
        choices = ["Option A", "Option B", "Option C"]
        result = _interactive_select("Choose:", choices)
        
        assert result == "Option C"
        mock_ui.success.assert_called_with("Selected: Option C")
    
    @patch('sys.stdout')
    @patch('mcp_cli.ui.prompts._get_key')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_interactive_select_number_selection(self, mock_get_theme, mock_ui, mock_get_key, mock_stdout):
        """Test interactive selection with number key."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        # Simulate pressing '2'
        mock_get_key.return_value = '2'
        
        choices = ["Option A", "Option B", "Option C"]
        result = _interactive_select("Choose:", choices)
        
        assert result == "Option B"
    
    @patch('sys.stdout')
    @patch('mcp_cli.ui.prompts._get_key')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_interactive_select_quit(self, mock_get_theme, mock_ui, mock_get_key, mock_stdout):
        """Test quitting interactive selection."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        mock_get_key.return_value = 'q'
        
        with pytest.raises(KeyboardInterrupt):
            _interactive_select("Choose:", ["A", "B"])


@pytest.mark.skip(reason="_interactive_multi_select no longer exposed after chuk-term migration")
class TestInteractiveMultiSelect:
    """Test _interactive_multi_select function."""
    
    @patch('sys.stdout')
    @patch('time.sleep')  # Mock sleep to speed up tests
    @patch('mcp_cli.ui.prompts._get_key')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_interactive_multi_select_basic(self, mock_get_theme, mock_ui, mock_get_key, mock_sleep, mock_stdout):
        """Test basic interactive multi-selection."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        # Simulate: space (select first), down, space (select second), enter
        mock_get_key.side_effect = ['space', 'down', 'space', 'enter']
        
        choices = ["Option A", "Option B", "Option C"]
        result = _interactive_multi_select("Choose multiple:", choices)
        
        # Check that both options were selected (order doesn't matter)
        assert set(result) == {"Option A", "Option B"}
        # Check that success message was shown with both items
        success_calls = [call for call in mock_ui.success.call_args_list]
        assert len(success_calls) > 0
        success_msg = str(success_calls[0])
        assert "Option A" in success_msg and "Option B" in success_msg
    
    @patch('sys.stdout')
    @patch('time.sleep')
    @patch('mcp_cli.ui.prompts._get_key')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_interactive_multi_select_all_none(self, mock_get_theme, mock_ui, mock_get_key, mock_sleep, mock_stdout):
        """Test select all and none shortcuts."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        # Simulate: 'a' (select all), 'n' (select none), 'enter'
        mock_get_key.side_effect = ['a', 'n', 'enter']
        
        choices = ["Option A", "Option B"]
        result = _interactive_multi_select("Choose:", choices)
        
        assert result == []  # None selected
        mock_ui.info.assert_called_with("No items selected")
    
    @patch('sys.stdout')
    @patch('time.sleep')
    @patch('mcp_cli.ui.prompts._get_key')
    @patch('mcp_cli.ui.prompts.ui')
    @patch('mcp_cli.ui.prompts.get_theme')
    def test_interactive_multi_select_constraints(self, mock_get_theme, mock_ui, mock_get_key, mock_sleep, mock_stdout):
        """Test multi-selection with constraints."""
        mock_theme = Mock()
        mock_theme.name = "default"
        mock_get_theme.return_value = mock_theme
        
        # Try to confirm with too few, then select enough and confirm
        mock_get_key.side_effect = ['enter', 'space', 'down', 'space', 'enter']
        
        choices = ["A", "B", "C"]
        result = _interactive_multi_select(
            "Choose:",
            choices,
            min_selections=2
        )
        
        assert len(result) >= 2
        mock_ui.warning.assert_called_with("Please select at least 2 items")


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset module state after each test."""
    # Remove msvcrt from sys.modules if it was added during test
    if 'msvcrt' in sys.modules and sys.platform != 'win32':
        del sys.modules['msvcrt']
    yield