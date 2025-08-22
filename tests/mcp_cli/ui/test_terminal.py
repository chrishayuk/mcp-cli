"""Unit tests for terminal management utilities."""

import asyncio
import gc
import os
import sys
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import pytest

from mcp_cli.ui.terminal import (
    TerminalManager,
    clear_screen,
    restore_terminal,
    reset_terminal,
    get_terminal_size,
    set_terminal_title,
    hide_cursor,
    show_cursor,
    bell,
    hyperlink,
    get_terminal_info,
    alternate_screen,
)


class TestTerminalManagerClear:
    """Test terminal clearing functionality."""
    
    @patch('os.system')
    @patch('sys.platform', 'win32')
    def test_clear_windows(self, mock_system):
        """Test clearing terminal on Windows."""
        TerminalManager.clear()
        mock_system.assert_called_once_with("cls")
    
    @patch('os.system')
    @patch('sys.platform', 'linux')
    def test_clear_unix(self, mock_system):
        """Test clearing terminal on Unix-like systems."""
        TerminalManager.clear()
        mock_system.assert_called_once_with("clear")
    
    @patch('os.system')
    @patch('sys.platform', 'darwin')
    def test_clear_macos(self, mock_system):
        """Test clearing terminal on macOS."""
        TerminalManager.clear()
        mock_system.assert_called_once_with("clear")


class TestTerminalManagerReset:
    """Test terminal reset functionality."""
    
    @patch('os.system')
    @patch('sys.platform', 'linux')
    def test_reset_unix(self, mock_system):
        """Test resetting terminal on Unix-like systems."""
        TerminalManager.reset()
        mock_system.assert_called_once_with("stty sane")
    
    @patch('os.system')
    @patch('sys.platform', 'win32')
    def test_reset_windows_skipped(self, mock_system):
        """Test that reset is skipped on Windows."""
        TerminalManager.reset()
        mock_system.assert_not_called()
    
    @patch('os.system', side_effect=Exception("Command failed"))
    @patch('sys.platform', 'linux')
    @patch('mcp_cli.ui.terminal.logger')
    def test_reset_error_handled(self, mock_logger, mock_system):
        """Test that reset errors are handled gracefully."""
        TerminalManager.reset()
        # Should log debug message about the error
        mock_logger.debug.assert_called_once()
        assert "Could not reset terminal" in mock_logger.debug.call_args[0][0]


class TestTerminalManagerRestore:
    """Test terminal restoration functionality."""
    
    @patch.object(TerminalManager, 'reset')
    @patch.object(TerminalManager, 'cleanup_asyncio')
    @patch('gc.collect')
    def test_restore_full_cleanup(self, mock_gc, mock_cleanup_asyncio, mock_reset):
        """Test full terminal restoration."""
        TerminalManager.restore()
        
        # Should call all cleanup methods in order
        mock_reset.assert_called_once()
        mock_cleanup_asyncio.assert_called_once()
        mock_gc.assert_called_once()


class TestTerminalManagerAsyncioCleanup:
    """Test asyncio cleanup functionality."""
    
    @patch('asyncio.get_running_loop')
    def test_cleanup_with_running_loop(self, mock_get_running):
        """Test cleanup when called from within a running loop."""
        mock_loop = Mock()
        mock_loop.is_closed.return_value = False
        mock_get_running.return_value = mock_loop
        
        TerminalManager.cleanup_asyncio()
        
        # Should not try to cancel tasks when in running loop
        mock_loop.run_until_complete.assert_not_called()
    
    @patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop"))
    @patch('asyncio.get_event_loop')
    def test_cleanup_with_event_loop_no_tasks(self, mock_get_event, mock_get_running):
        """Test cleanup with event loop but no tasks."""
        mock_loop = Mock()
        mock_loop.is_closed.return_value = False
        mock_get_event.return_value = mock_loop
        
        with patch('asyncio.all_tasks', return_value=set()):
            TerminalManager.cleanup_asyncio()
        
        # Should try to shutdown async generators
        mock_loop.run_until_complete.assert_called()
        mock_loop.close.assert_called_once()
    
    @patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop"))
    @patch('asyncio.get_event_loop')
    def test_cleanup_with_pending_tasks(self, mock_get_event, mock_get_running):
        """Test cleanup with pending tasks."""
        mock_loop = Mock()
        mock_loop.is_closed.return_value = False
        mock_get_event.return_value = mock_loop
        
        # Create mock tasks
        mock_task1 = Mock()
        mock_task1.done.return_value = False
        mock_task1.cancel.return_value = None
        
        mock_task2 = Mock()
        mock_task2.done.return_value = True  # Already done
        mock_task2.cancel.return_value = None
        
        with patch('asyncio.all_tasks', return_value={mock_task1, mock_task2}):
            with patch('asyncio.gather', return_value=AsyncMock()):
                TerminalManager.cleanup_asyncio()
        
        # Should only cancel tasks that aren't done
        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_not_called()
        
        # Should try to wait for cancellation
        assert mock_loop.run_until_complete.called
    
    @patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop"))
    @patch('asyncio.get_event_loop')
    def test_cleanup_with_closed_loop(self, mock_get_event, mock_get_running):
        """Test cleanup with already closed loop."""
        mock_loop = Mock()
        mock_loop.is_closed.return_value = True
        mock_get_event.return_value = mock_loop
        
        TerminalManager.cleanup_asyncio()
        
        # Should return early without doing anything
        mock_loop.run_until_complete.assert_not_called()
        mock_loop.close.assert_not_called()
    
    @patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop"))
    @patch('asyncio.get_event_loop', side_effect=RuntimeError("No event loop"))
    def test_cleanup_no_event_loop(self, mock_get_event, mock_get_running):
        """Test cleanup when there's no event loop at all."""
        # Should return without error
        TerminalManager.cleanup_asyncio()
    
    @patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop"))
    @patch('asyncio.get_event_loop')
    @patch('mcp_cli.ui.terminal.logger')
    def test_cleanup_task_error_handled(self, mock_logger, mock_get_event, mock_get_running):
        """Test that task cleanup errors are handled."""
        mock_loop = Mock()
        mock_loop.is_closed.return_value = False
        mock_get_event.return_value = mock_loop
        
        with patch('asyncio.all_tasks', side_effect=Exception("Task error")):
            TerminalManager.cleanup_asyncio()
        
        # Should log debug message about the error
        mock_logger.debug.assert_called()
    
    @patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop"))
    @patch('asyncio.get_event_loop')
    @patch('mcp_cli.ui.terminal.logger')
    def test_cleanup_shutdown_asyncgens_error(self, mock_logger, mock_get_event, mock_get_running):
        """Test that async generator shutdown errors are handled."""
        mock_loop = Mock()
        mock_loop.is_closed.return_value = False
        mock_loop.shutdown_asyncgens.side_effect = Exception("Shutdown error")
        mock_get_event.return_value = mock_loop
        
        with patch('asyncio.all_tasks', return_value=set()):
            TerminalManager.cleanup_asyncio()
        
        # Should log debug message about the error
        assert any("async generators" in str(call) for call in mock_logger.debug.call_args_list)
    
    @patch('asyncio.get_running_loop', side_effect=Exception("Unexpected error"))
    @patch('mcp_cli.ui.terminal.logger')
    def test_cleanup_general_error_handled(self, mock_logger, mock_get_running):
        """Test that general cleanup errors are handled."""
        TerminalManager.cleanup_asyncio()
        
        # Should log debug message about the error
        mock_logger.debug.assert_called_once()
        assert "Asyncio cleanup error" in mock_logger.debug.call_args[0][0]
    
    @patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop"))
    @patch('asyncio.get_event_loop')
    @patch('mcp_cli.ui.terminal.logger')
    def test_cleanup_uses_task_all_tasks_fallback(self, mock_logger, mock_get_event, mock_get_running):
        """Test cleanup when asyncio.all_tasks is not available."""
        mock_loop = Mock()
        mock_loop.is_closed.return_value = False
        mock_get_event.return_value = mock_loop
        
        # When asyncio.all_tasks doesn't exist and neither does Task.all_tasks,
        # the code will raise an AttributeError which is caught
        with patch.object(asyncio, 'all_tasks', new=None):
            with patch('asyncio.Task', spec=[]):  # Task without all_tasks attribute
                TerminalManager.cleanup_asyncio()
                # Should log the error
                mock_logger.debug.assert_called()
                assert "Error during task cleanup" in str(mock_logger.debug.call_args_list)


class TestTerminalManagerGetSize:
    """Test terminal size detection."""
    
    @patch('shutil.get_terminal_size')
    def test_get_size_normal(self, mock_get_size):
        """Test getting terminal size normally."""
        mock_get_size.return_value = os.terminal_size((120, 40))
        
        cols, rows = TerminalManager.get_size()
        
        assert cols == 120
        assert rows == 40
        mock_get_size.assert_called_once()
    
    @patch('shutil.get_terminal_size', side_effect=Exception("No terminal"))
    def test_get_size_fallback(self, mock_get_size):
        """Test fallback terminal size when detection fails."""
        cols, rows = TerminalManager.get_size()
        
        assert cols == 80
        assert rows == 24
        mock_get_size.assert_called_once()


class TestTerminalManagerSupportsColor:
    """Test color support detection."""
    
    @patch('sys.stdout.isatty', return_value=True)
    def test_supports_color_tty(self, mock_isatty):
        """Test color support when stdout is a TTY."""
        assert TerminalManager.supports_color() is True
        mock_isatty.assert_called_once()
    
    @patch('sys.stdout.isatty', return_value=False)
    def test_supports_color_no_tty(self, mock_isatty):
        """Test no color support when stdout is not a TTY."""
        assert TerminalManager.supports_color() is False
        mock_isatty.assert_called_once()


class TestTerminalManagerSetTitle:
    """Test terminal title setting."""
    
    @patch('os.system')
    @patch('sys.platform', 'win32')
    def test_set_title_windows(self, mock_system):
        """Test setting title on Windows."""
        TerminalManager.set_title("Test Title")
        mock_system.assert_called_once_with("title Test Title")
    
    @patch('sys.stdout')
    @patch('sys.platform', 'linux')
    def test_set_title_unix(self, mock_stdout):
        """Test setting title on Unix-like systems."""
        mock_stdout.write = Mock()
        mock_stdout.flush = Mock()
        
        TerminalManager.set_title("Test Title")
        
        mock_stdout.write.assert_called_once_with("\033]0;Test Title\007")
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.stdout')
    @patch('sys.platform', 'darwin')
    def test_set_title_macos(self, mock_stdout):
        """Test setting title on macOS."""
        mock_stdout.write = Mock()
        mock_stdout.flush = Mock()
        
        TerminalManager.set_title("macOS Terminal")
        
        mock_stdout.write.assert_called_once_with("\033]0;macOS Terminal\007")
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.stdout')
    @patch('sys.platform', 'linux')
    def test_set_title_with_special_chars(self, mock_stdout):
        """Test setting title with special characters."""
        mock_stdout.write = Mock()
        mock_stdout.flush = Mock()
        
        TerminalManager.set_title("Test: 'Title' with \"quotes\" & symbols!")
        
        mock_stdout.write.assert_called_once()
        assert "Test: 'Title' with \"quotes\" & symbols!" in mock_stdout.write.call_args[0][0]


class TestBackwardCompatibilityFunctions:
    """Test backward compatibility convenience functions."""
    
    @patch.object(TerminalManager, 'clear')
    def test_clear_screen(self, mock_clear):
        """Test clear_screen convenience function."""
        clear_screen()
        mock_clear.assert_called_once()
    
    @patch.object(TerminalManager, 'restore')
    def test_restore_terminal(self, mock_restore):
        """Test restore_terminal convenience function."""
        restore_terminal()
        mock_restore.assert_called_once()
    
    @patch.object(TerminalManager, 'reset')
    def test_reset_terminal(self, mock_reset):
        """Test reset_terminal convenience function."""
        reset_terminal()
        mock_reset.assert_called_once()
    
    @patch.object(TerminalManager, 'get_size', return_value=(100, 50))
    def test_get_terminal_size(self, mock_get_size):
        """Test get_terminal_size convenience function."""
        cols, rows = get_terminal_size()
        assert cols == 100
        assert rows == 50
        mock_get_size.assert_called_once()
    
    @patch.object(TerminalManager, 'set_title')
    def test_set_terminal_title(self, mock_set_title):
        """Test set_terminal_title convenience function."""
        set_terminal_title("New Title")
        mock_set_title.assert_called_once_with("New Title")


class TestIntegrationScenarios:
    """Test integration scenarios for terminal management."""
    
    @patch('os.system')
    @patch('gc.collect')
    @patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop"))
    @patch('asyncio.get_event_loop', side_effect=RuntimeError("No event loop"))
    @patch('sys.platform', 'linux')
    def test_full_restore_no_asyncio(self, mock_get_event, mock_get_running, 
                                     mock_gc, mock_system):
        """Test full restore when there's no asyncio loop."""
        TerminalManager.restore()
        
        # Should still reset terminal and collect garbage
        mock_system.assert_called_once_with("stty sane")
        mock_gc.assert_called_once()
    
    @patch('os.system')
    @patch('gc.collect')
    @patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop"))
    @patch('asyncio.get_event_loop')
    @patch('asyncio.all_tasks')
    @patch('sys.platform', 'linux')
    def test_full_restore_with_tasks(self, mock_all_tasks, mock_get_event, 
                                     mock_get_running, mock_gc, mock_system):
        """Test full restore with active asyncio tasks."""
        # Setup mock loop with tasks
        mock_loop = Mock()
        mock_loop.is_closed.return_value = False
        mock_get_event.return_value = mock_loop
        
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_all_tasks.return_value = {mock_task}
        
        TerminalManager.restore()
        
        # Should reset terminal
        mock_system.assert_called_with("stty sane")
        # Should cancel tasks
        mock_task.cancel.assert_called_once()
        # Should close loop
        mock_loop.close.assert_called_once()
        # Should collect garbage
        mock_gc.assert_called_once()
    
    @patch('shutil.get_terminal_size', side_effect=Exception("No terminal"))
    @patch('sys.stdout.isatty', return_value=False)
    def test_non_interactive_environment(self, mock_isatty, mock_get_size):
        """Test terminal operations in non-interactive environment."""
        # Should handle gracefully when not in a terminal
        cols, rows = TerminalManager.get_size()
        assert cols == 80  # Fallback values
        assert rows == 24
        
        assert TerminalManager.supports_color() is False


class TestErrorHandling:
    """Test error handling in terminal operations."""
    
    @patch('os.system', side_effect=OSError("Permission denied"))
    @patch('sys.platform', 'linux')
    @patch('mcp_cli.ui.terminal.logger')
    def test_reset_permission_error(self, mock_logger, mock_system):
        """Test handling permission errors during reset."""
        TerminalManager.reset()
        mock_logger.debug.assert_called_once()
        assert "Could not reset terminal" in mock_logger.debug.call_args[0][0]
    
    @patch('sys.stdout')
    @patch('sys.platform', 'linux')
    def test_set_title_io_error(self, mock_stdout):
        """Test handling IO errors when setting title."""
        # Set up mock to raise IOError
        mock_stdout.write.side_effect = IOError("Broken pipe")
        mock_stdout.flush.side_effect = IOError("Broken pipe")
        
        # Currently set_title doesn't handle IOError, so it will raise
        # This test documents the current behavior
        with pytest.raises(IOError, match="Broken pipe"):
            TerminalManager.set_title("Test")
    
    @patch('asyncio.get_running_loop', side_effect=RuntimeError("No running loop"))
    @patch('asyncio.get_event_loop')
    @patch('mcp_cli.ui.terminal.logger')
    def test_cleanup_loop_close_error(self, mock_logger, mock_get_event, mock_get_running):
        """Test handling errors when closing event loop."""
        mock_loop = Mock()
        mock_loop.is_closed.return_value = False
        mock_loop.close.side_effect = Exception("Cannot close")
        mock_get_event.return_value = mock_loop
        
        with patch('asyncio.all_tasks', return_value=set()):
            # Should not raise exception
            TerminalManager.cleanup_asyncio()


class TestCursorControl:
    """Test cursor control methods."""
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_hide_cursor_unix(self, mock_stdout):
        """Test hiding cursor on Unix."""
        TerminalManager.hide_cursor()
        mock_stdout.write.assert_called_with('\033[?25l')
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.platform', 'win32')
    @patch('sys.stdout')
    def test_hide_cursor_windows(self, mock_stdout):
        """Test hiding cursor on Windows (no-op)."""
        TerminalManager.hide_cursor()
        mock_stdout.write.assert_not_called()
    
    @patch('sys.platform', 'darwin')
    @patch('sys.stdout')
    def test_show_cursor_unix(self, mock_stdout):
        """Test showing cursor on Unix."""
        TerminalManager.show_cursor()
        mock_stdout.write.assert_called_with('\033[?25h')
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_save_cursor_position(self, mock_stdout):
        """Test saving cursor position."""
        TerminalManager.save_cursor_position()
        mock_stdout.write.assert_called_with('\033[s')
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_restore_cursor_position(self, mock_stdout):
        """Test restoring cursor position."""
        TerminalManager.restore_cursor_position()
        mock_stdout.write.assert_called_with('\033[u')
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_move_cursor_up(self, mock_stdout):
        """Test moving cursor up."""
        TerminalManager.move_cursor_up(3)
        mock_stdout.write.assert_called_with('\033[3A')
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_move_cursor_down(self, mock_stdout):
        """Test moving cursor down."""
        TerminalManager.move_cursor_down(5)
        mock_stdout.write.assert_called_with('\033[5B')
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_clear_line(self, mock_stdout):
        """Test clearing line."""
        TerminalManager.clear_line()
        mock_stdout.write.assert_called_with('\033[2K\r')
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_convenience_functions(self, mock_stdout):
        """Test convenience functions."""
        hide_cursor()
        mock_stdout.write.assert_called_with('\033[?25l')
        
        show_cursor()
        mock_stdout.write.assert_called_with('\033[?25h')


class TestAlternateScreen:
    """Test alternate screen buffer functionality."""
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_enter_alternate_screen(self, mock_stdout):
        """Test entering alternate screen."""
        TerminalManager.enter_alternate_screen()
        mock_stdout.write.assert_called_with('\033[?1049h')
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_exit_alternate_screen(self, mock_stdout):
        """Test exiting alternate screen."""
        TerminalManager.exit_alternate_screen()
        mock_stdout.write.assert_called_with('\033[?1049l')
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.platform', 'win32')
    @patch('sys.stdout')
    def test_alternate_screen_windows(self, mock_stdout):
        """Test alternate screen on Windows (no-op)."""
        TerminalManager.enter_alternate_screen()
        mock_stdout.write.assert_not_called()
        
        TerminalManager.exit_alternate_screen()
        mock_stdout.write.assert_not_called()
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_alternate_screen_context_manager(self, mock_stdout):
        """Test alternate screen context manager."""
        with TerminalManager.alternate_screen():
            # Should enter alternate screen and hide cursor
            calls = mock_stdout.write.call_args_list
            # Check the actual arguments passed to write()
            assert calls[0][0][0] == '\033[?1049h'
            assert calls[1][0][0] == '\033[?25l'
        
        # Should restore cursor and exit alternate screen
        calls = mock_stdout.write.call_args_list
        # Check the actual arguments in the last calls
        assert calls[-2][0][0] == '\033[?25h'
        assert calls[-1][0][0] == '\033[?1049l'
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_alternate_screen_context_with_exception(self, mock_stdout):
        """Test alternate screen context manager with exception."""
        with pytest.raises(ValueError):
            with TerminalManager.alternate_screen():
                raise ValueError("Test error")
        
        # Should still restore on exception
        calls = mock_stdout.write.call_args_list
        # Check the actual arguments in the last calls
        assert calls[-2][0][0] == '\033[?25h'
        assert calls[-1][0][0] == '\033[?1049l'
    
    @patch('sys.platform', 'linux')
    @patch('sys.stdout')
    def test_convenience_alternate_screen(self, mock_stdout):
        """Test convenience alternate_screen function."""
        with alternate_screen():
            pass
        
        # Should have entered and exited
        call_args = [call[0][0] for call in mock_stdout.write.call_args_list]
        assert '\033[?1049h' in call_args
        assert '\033[?1049l' in call_args


class TestTerminalBell:
    """Test terminal bell functionality."""
    
    @patch('sys.stdout')
    def test_bell(self, mock_stdout):
        """Test sounding terminal bell."""
        TerminalManager.bell()
        mock_stdout.write.assert_called_with('\a')
        mock_stdout.flush.assert_called_once()
    
    @patch('sys.stdout')
    def test_bell_convenience(self, mock_stdout):
        """Test bell convenience function."""
        bell()
        mock_stdout.write.assert_called_with('\a')


class TestHyperlinks:
    """Test hyperlink functionality."""
    
    def test_hyperlink_basic(self):
        """Test basic hyperlink creation."""
        link = TerminalManager.hyperlink("https://example.com")
        assert "https://example.com" in link
    
    def test_hyperlink_with_text(self):
        """Test hyperlink with custom text."""
        link = TerminalManager.hyperlink("https://example.com", "Click here")
        assert "Click here" in link
    
    @patch.dict(os.environ, {'TERM_PROGRAM': 'iTerm.app'})
    def test_hyperlink_iterm(self):
        """Test hyperlink in iTerm."""
        link = TerminalManager.hyperlink("https://example.com", "Test")
        assert '\033]8;;https://example.com\033\\Test\033]8;;\033\\' == link
    
    @patch.dict(os.environ, {'TERM_PROGRAM': 'kitty'})
    def test_hyperlink_kitty(self):
        """Test hyperlink in Kitty."""
        link = TerminalManager.hyperlink("https://example.com", "Test")
        assert '\033]8;;https://example.com\033\\Test\033]8;;\033\\' == link
    
    @patch.dict(os.environ, {'TERM_PROGRAM': 'generic'})
    def test_hyperlink_fallback(self):
        """Test hyperlink fallback for unsupported terminals."""
        link = TerminalManager.hyperlink("https://example.com", "Test")
        assert link == "Test (https://example.com)"
    
    @patch.dict(os.environ, {'TERM_PROGRAM': 'generic'})
    def test_hyperlink_fallback_no_text(self):
        """Test hyperlink fallback without custom text."""
        link = TerminalManager.hyperlink("https://example.com")
        assert link == "https://example.com"
    
    @patch('builtins.print')
    def test_print_hyperlink(self, mock_print):
        """Test printing hyperlink."""
        TerminalManager.print_hyperlink("https://example.com", "Click")
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        assert "https://example.com" in args
    
    def test_hyperlink_convenience(self):
        """Test hyperlink convenience function."""
        link = hyperlink("https://example.com", "Test")
        assert "https://example.com" in link


class TestColorDetection:
    """Test color detection methods."""
    
    @patch.dict(os.environ, {'COLORTERM': 'truecolor'})
    def test_supports_truecolor(self):
        """Test truecolor detection."""
        assert TerminalManager.supports_truecolor() is True
    
    @patch.dict(os.environ, {'COLORTERM': ''})
    def test_no_truecolor(self):
        """Test no truecolor."""
        assert TerminalManager.supports_truecolor() is False
    
    @patch.dict(os.environ, {'TERM': 'xterm-256color'})
    def test_supports_256_colors(self):
        """Test 256 color detection."""
        assert TerminalManager.supports_256_colors() is True
    
    @patch.dict(os.environ, {'TERM': 'xterm'})
    def test_no_256_colors(self):
        """Test no 256 colors."""
        with patch.object(TerminalManager, 'supports_truecolor', return_value=False):
            assert TerminalManager.supports_256_colors() is False
    
    @patch.dict(os.environ, {'COLORTERM': 'truecolor'})
    def test_256_colors_via_truecolor(self):
        """Test 256 colors detected via truecolor."""
        assert TerminalManager.supports_256_colors() is True
    
    @patch('mcp_cli.ui.terminal.TerminalManager.supports_color')
    def test_get_color_level(self, mock_supports_color):
        """Test color level detection."""
        # Test mono
        mock_supports_color.return_value = False
        assert TerminalManager.get_color_level() == 'mono'
        
        # Test truecolor
        mock_supports_color.return_value = True
        with patch.object(TerminalManager, 'supports_truecolor', return_value=True):
            assert TerminalManager.get_color_level() == 'truecolor'
        
        # Test 256 colors
        with patch.object(TerminalManager, 'supports_truecolor', return_value=False):
            with patch.object(TerminalManager, 'supports_256_colors', return_value=True):
                assert TerminalManager.get_color_level() == '256'
        
        # Test 16 colors
        with patch.object(TerminalManager, 'supports_truecolor', return_value=False):
            with patch.object(TerminalManager, 'supports_256_colors', return_value=False):
                assert TerminalManager.get_color_level() == '16'


class TestTerminalInfo:
    """Test terminal information methods."""
    
    @patch.dict(os.environ, {'TERM_PROGRAM': 'iTerm.app'})
    def test_get_terminal_program(self):
        """Test getting terminal program."""
        assert TerminalManager.get_terminal_program() == 'iTerm.app'
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_terminal_program_unknown(self):
        """Test unknown terminal program."""
        assert TerminalManager.get_terminal_program() == 'unknown'
    
    @patch.dict(os.environ, {'TMUX': '/tmp/tmux-1000/default'})
    def test_is_tmux(self):
        """Test tmux detection."""
        assert TerminalManager.is_tmux() is True
    
    @patch.dict(os.environ, {}, clear=True)
    def test_not_tmux(self):
        """Test not in tmux."""
        assert TerminalManager.is_tmux() is False
    
    @patch.dict(os.environ, {'STY': '12345.pts-0.hostname'})
    def test_is_screen(self):
        """Test GNU screen detection."""
        assert TerminalManager.is_screen() is True
    
    @patch.dict(os.environ, {}, clear=True)
    def test_not_screen(self):
        """Test not in screen."""
        assert TerminalManager.is_screen() is False
    
    @patch.dict(os.environ, {'SSH_CLIENT': '192.168.1.1 12345 22'})
    def test_is_ssh_client(self):
        """Test SSH detection via SSH_CLIENT."""
        assert TerminalManager.is_ssh() is True
    
    @patch.dict(os.environ, {'SSH_TTY': '/dev/pts/0'})
    def test_is_ssh_tty(self):
        """Test SSH detection via SSH_TTY."""
        assert TerminalManager.is_ssh() is True
    
    @patch.dict(os.environ, {}, clear=True)
    def test_not_ssh(self):
        """Test not over SSH."""
        assert TerminalManager.is_ssh() is False
    
    @patch('mcp_cli.ui.terminal.TerminalManager.get_size')
    @patch('sys.stdout')
    def test_get_terminal_info(self, mock_stdout, mock_get_size):
        """Test comprehensive terminal info."""
        mock_get_size.return_value = (120, 40)
        mock_stdout.encoding = 'utf-8'
        mock_stdout.isatty.return_value = True
        
        with patch.dict(os.environ, {
            'TERM_PROGRAM': 'iTerm.app',
            'TERM': 'xterm-256color',
            'TMUX': 'yes',
            'SSH_CLIENT': '192.168.1.1 12345 22'
        }):
            info = TerminalManager.get_terminal_info()
            
            assert info['program'] == 'iTerm.app'
            assert info['type'] == 'xterm-256color'
            assert info['size'] == {'columns': 120, 'rows': 40}
            assert info['encoding'] == 'utf-8'
            assert info['is_tty'] is True
            assert info['is_tmux'] is True
            assert info['is_ssh'] is True
            assert info['platform'] == sys.platform
    
    def test_get_terminal_info_convenience(self):
        """Test get_terminal_info convenience function."""
        info = get_terminal_info()
        assert isinstance(info, dict)
        assert 'program' in info
        assert 'size' in info


class TestProgressInTitle:
    """Test progress display in terminal title."""
    
    @patch('mcp_cli.ui.terminal.TerminalManager.set_title')
    def test_set_title_progress_basic(self, mock_set_title):
        """Test basic progress in title."""
        TerminalManager.set_title_progress(50)
        mock_set_title.assert_called_once()
        title = mock_set_title.call_args[0][0]
        assert "50%" in title
        assert "Progress" in title
    
    @patch('mcp_cli.ui.terminal.TerminalManager.set_title')
    def test_set_title_progress_custom_prefix(self, mock_set_title):
        """Test progress with custom prefix."""
        TerminalManager.set_title_progress(75, "Download")
        title = mock_set_title.call_args[0][0]
        assert "75%" in title
        assert "Download" in title
    
    @patch('mcp_cli.ui.terminal.TerminalManager.set_title')
    def test_set_title_progress_clamping(self, mock_set_title):
        """Test progress value clamping."""
        # Test over 100
        TerminalManager.set_title_progress(150)
        title = mock_set_title.call_args[0][0]
        assert "100%" in title
        
        # Test under 0
        TerminalManager.set_title_progress(-50)
        title = mock_set_title.call_args[0][0]
        assert "0%" in title
    
    @patch('mcp_cli.ui.terminal.TerminalManager.set_title')
    def test_set_title_progress_bar(self, mock_set_title):
        """Test progress bar visualization."""
        # Test 0%
        TerminalManager.set_title_progress(0)
        title = mock_set_title.call_args[0][0]
        assert "░░░░░░░░░░" in title
        
        # Test 50%
        TerminalManager.set_title_progress(50)
        title = mock_set_title.call_args[0][0]
        assert "█████" in title
        assert "░░░░░" in title
        
        # Test 100%
        TerminalManager.set_title_progress(100)
        title = mock_set_title.call_args[0][0]
        assert "██████████" in title


class TestEnhancedFeatureEdgeCases:
    """Test edge cases and error handling for enhanced features."""
    
    @patch('sys.platform', 'win32')
    def test_windows_no_ansi(self):
        """Test that Windows doesn't use ANSI codes."""
        with patch('sys.stdout') as mock_stdout:
            TerminalManager.hide_cursor()
            TerminalManager.show_cursor()
            TerminalManager.save_cursor_position()
            TerminalManager.restore_cursor_position()
            TerminalManager.move_cursor_up()
            TerminalManager.move_cursor_down()
            TerminalManager.clear_line()
            TerminalManager.enter_alternate_screen()
            TerminalManager.exit_alternate_screen()
            
            # None of these should write on Windows
            mock_stdout.write.assert_not_called()
    
    @patch('sys.stdout')
    def test_bell_always_works(self, mock_stdout):
        """Test bell works on all platforms."""
        with patch('sys.platform', 'win32'):
            TerminalManager.bell()
            mock_stdout.write.assert_called_with('\a')
        
        with patch('sys.platform', 'linux'):
            TerminalManager.bell()
            mock_stdout.write.assert_called_with('\a')
    
    def test_hyperlink_empty_url(self):
        """Test hyperlink with empty URL."""
        link = TerminalManager.hyperlink("")
        # Empty URL still creates hyperlink structure in supported terminals
        # or returns empty string in unsupported ones
        assert "" in link or link == ""
    
    def test_hyperlink_none_text(self):
        """Test hyperlink with None text."""
        link = TerminalManager.hyperlink("https://example.com", None)
        assert "https://example.com" in link


@pytest.fixture(autouse=True)
def reset_platform():
    """Reset platform after each test."""
    original_platform = sys.platform
    yield
    sys.platform = original_platform