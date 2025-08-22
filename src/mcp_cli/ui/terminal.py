# src/mcp_cli/ui/terminal.py
"""
Terminal management utilities.

Handles terminal state, cleanup, and cross-platform terminal operations.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TerminalManager:
    """Manages terminal state and cleanup."""
    
    @staticmethod
    def clear() -> None:
        """Clear the terminal screen (cross-platform)."""
        if sys.platform == "win32":
            os.system("cls")
        else:
            os.system("clear")
    
    @staticmethod
    def reset() -> None:
        """Reset terminal to sane state (Unix-like systems)."""
        if sys.platform != "win32":
            try:
                os.system("stty sane")
            except Exception as e:
                logger.debug(f"Could not reset terminal: {e}")
    
    @staticmethod
    def restore() -> None:
        """
        Fully restore terminal and clean up resources.
        
        This should be called on application exit.
        """
        # Reset terminal settings
        TerminalManager.reset()
        
        # Clean up asyncio
        TerminalManager.cleanup_asyncio()
        
        # Force garbage collection
        gc.collect()
    
    @staticmethod
    def cleanup_asyncio() -> None:
        """Clean up asyncio resources gracefully."""
        try:
            # Try to get the running loop first
            try:
                loop = asyncio.get_running_loop()
                is_running = True
            except RuntimeError:
                # No running loop, try to get the event loop
                try:
                    loop = asyncio.get_event_loop()
                    is_running = False
                except RuntimeError:
                    # No event loop at all
                    return
            
            if loop.is_closed():
                return
            
            # Only cancel tasks if we're not in a running loop
            # (if we're in a running loop, we're likely being called from within asyncio)
            if not is_running:
                # Get all tasks
                try:
                    if hasattr(asyncio, 'all_tasks'):
                        pending = asyncio.all_tasks(loop)
                    else:
                        pending = asyncio.Task.all_tasks(loop)
                    
                    # Only cancel tasks that aren't done
                    tasks = [t for t in pending if not t.done()]
                    
                    for task in tasks:
                        task.cancel()
                    
                    # Give tasks a chance to cancel gracefully
                    if tasks:
                        try:
                            loop.run_until_complete(
                                asyncio.gather(*tasks, return_exceptions=True)
                            )
                        except Exception:
                            # Tasks didn't cancel cleanly, but that's okay
                            pass
                    
                    # Shutdown async generators
                    try:
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    except Exception as e:
                        logger.debug(f"Error shutting down async generators: {e}")
                    
                    # Close the loop
                    try:
                        loop.close()
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(f"Error during task cleanup: {e}")
            
        except Exception as exc:
            logger.debug(f"Asyncio cleanup error: {exc}")
    
    @staticmethod
    def get_size() -> tuple[int, int]:
        """
        Get terminal size.
        
        Returns:
            Tuple of (columns, rows)
        """
        try:
            import shutil
            return shutil.get_terminal_size()
        except Exception:
            return (80, 24)  # Default fallback
    
    @staticmethod
    def supports_color() -> bool:
        """Check if terminal supports color output."""
        return sys.stdout.isatty()
    
    @staticmethod
    def set_title(title: str) -> None:
        """
        Set terminal window title.
        
        Args:
            title: New terminal title
        """
        if sys.platform == "win32":
            os.system(f"title {title}")
        else:
            sys.stdout.write(f"\033]0;{title}\007")
            sys.stdout.flush()
    
    # ─────────────────────────── Cursor Control ─────────────────────────
    
    @staticmethod
    def hide_cursor() -> None:
        """Hide the terminal cursor."""
        if sys.platform != "win32":
            sys.stdout.write('\033[?25l')
            sys.stdout.flush()
    
    @staticmethod
    def show_cursor() -> None:
        """Show the terminal cursor."""
        if sys.platform != "win32":
            sys.stdout.write('\033[?25h')
            sys.stdout.flush()
    
    @staticmethod
    def save_cursor_position() -> None:
        """Save current cursor position."""
        if sys.platform != "win32":
            sys.stdout.write('\033[s')
            sys.stdout.flush()
    
    @staticmethod
    def restore_cursor_position() -> None:
        """Restore saved cursor position."""
        if sys.platform != "win32":
            sys.stdout.write('\033[u')
            sys.stdout.flush()
    
    @staticmethod
    def move_cursor_up(lines: int = 1) -> None:
        """Move cursor up by n lines."""
        if sys.platform != "win32":
            sys.stdout.write(f'\033[{lines}A')
            sys.stdout.flush()
    
    @staticmethod
    def move_cursor_down(lines: int = 1) -> None:
        """Move cursor down by n lines."""
        if sys.platform != "win32":
            sys.stdout.write(f'\033[{lines}B')
            sys.stdout.flush()
    
    @staticmethod
    def clear_line() -> None:
        """Clear current line."""
        if sys.platform != "win32":
            sys.stdout.write('\033[2K\r')
            sys.stdout.flush()
    
    # ─────────────────────────── Alternate Screen ───────────────────────
    
    @staticmethod
    def enter_alternate_screen() -> None:
        """
        Switch to alternate screen buffer.
        Useful for full-screen applications.
        """
        if sys.platform != "win32":
            sys.stdout.write('\033[?1049h')
            sys.stdout.flush()
    
    @staticmethod
    def exit_alternate_screen() -> None:
        """
        Return from alternate screen buffer.
        Restores previous screen content.
        """
        if sys.platform != "win32":
            sys.stdout.write('\033[?1049l')
            sys.stdout.flush()
    
    @staticmethod
    @contextmanager
    def alternate_screen():
        """
        Context manager for alternate screen.
        
        Usage:
            with TerminalManager.alternate_screen():
                # Your full-screen app here
                pass
        """
        try:
            TerminalManager.enter_alternate_screen()
            TerminalManager.hide_cursor()
            yield
        finally:
            TerminalManager.show_cursor()
            TerminalManager.exit_alternate_screen()
    
    # ─────────────────────────── Terminal Bell ──────────────────────────
    
    @staticmethod
    def bell() -> None:
        """Sound the terminal bell/beep."""
        sys.stdout.write('\a')
        sys.stdout.flush()
    
    # ─────────────────────────── Hyperlinks ─────────────────────────────
    
    @staticmethod
    def hyperlink(url: str, text: Optional[str] = None) -> str:
        """
        Create a clickable hyperlink (OSC 8).
        
        Supported in modern terminals like iTerm2, Kitty, etc.
        
        Args:
            url: The URL to link to
            text: Display text (defaults to URL)
            
        Returns:
            Formatted hyperlink string
        """
        if not text:
            text = url
        
        # Check if terminal likely supports hyperlinks
        term_program = os.environ.get('TERM_PROGRAM', '')
        if term_program in ('iTerm.app', 'Hyper', 'kitty', 'WezTerm'):
            # OSC 8 hyperlink format
            return f'\033]8;;{url}\033\\{text}\033]8;;\033\\'
        else:
            # Fallback to plain text
            return f"{text} ({url})" if text != url else url
    
    @staticmethod
    def print_hyperlink(url: str, text: Optional[str] = None) -> None:
        """Print a clickable hyperlink."""
        print(TerminalManager.hyperlink(url, text))
    
    # ─────────────────────────── Color Detection ────────────────────────
    
    @staticmethod
    def supports_truecolor() -> bool:
        """Check if terminal supports true color (24-bit RGB)."""
        return os.environ.get('COLORTERM') == 'truecolor'
    
    @staticmethod
    def supports_256_colors() -> bool:
        """Check if terminal supports 256 colors."""
        term = os.environ.get('TERM', '')
        return '256color' in term or TerminalManager.supports_truecolor()
    
    @staticmethod
    def get_color_level() -> str:
        """
        Get the color support level.
        
        Returns:
            'truecolor', '256', '16', or 'mono'
        """
        if not TerminalManager.supports_color():
            return 'mono'
        elif TerminalManager.supports_truecolor():
            return 'truecolor'
        elif TerminalManager.supports_256_colors():
            return '256'
        else:
            return '16'
    
    # ─────────────────────────── Terminal Info ──────────────────────────
    
    @staticmethod
    def get_terminal_program() -> str:
        """Get the terminal program name."""
        return os.environ.get('TERM_PROGRAM', 'unknown')
    
    @staticmethod
    def is_tmux() -> bool:
        """Check if running inside tmux."""
        return 'TMUX' in os.environ
    
    @staticmethod
    def is_screen() -> bool:
        """Check if running inside GNU screen."""
        return 'STY' in os.environ
    
    @staticmethod
    def is_ssh() -> bool:
        """Check if running over SSH."""
        return 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ
    
    @staticmethod
    def get_terminal_info() -> dict:
        """
        Get comprehensive terminal information.
        
        Returns:
            Dictionary with terminal details
        """
        cols, rows = TerminalManager.get_size()
        
        return {
            'program': TerminalManager.get_terminal_program(),
            'type': os.environ.get('TERM', 'unknown'),
            'size': {'columns': cols, 'rows': rows},
            'color_level': TerminalManager.get_color_level(),
            'supports_color': TerminalManager.supports_color(),
            'supports_256_colors': TerminalManager.supports_256_colors(),
            'supports_truecolor': TerminalManager.supports_truecolor(),
            'encoding': sys.stdout.encoding,
            'is_tty': sys.stdout.isatty(),
            'is_tmux': TerminalManager.is_tmux(),
            'is_screen': TerminalManager.is_screen(),
            'is_ssh': TerminalManager.is_ssh(),
            'platform': sys.platform,
        }
    
    # ─────────────────────────── Progress in Title ──────────────────────
    
    @staticmethod
    def set_title_progress(percent: int, prefix: str = "Progress") -> None:
        """
        Show progress in terminal title.
        
        Args:
            percent: Progress percentage (0-100)
            prefix: Text prefix for the title
        """
        # Ensure percent is in valid range
        percent = max(0, min(100, percent))
        
        # Create progress bar for title
        bar_length = 10
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        title = f"{prefix}: [{bar}] {percent}%"
        TerminalManager.set_title(title)


# Convenience functions for backward compatibility
def clear_screen() -> None:
    """Clear the terminal screen."""
    TerminalManager.clear()


def restore_terminal() -> None:
    """Restore terminal settings and clean up resources."""
    TerminalManager.restore()


def reset_terminal() -> None:
    """Reset terminal to sane state."""
    TerminalManager.reset()


def get_terminal_size() -> tuple[int, int]:
    """Get terminal size as (columns, rows)."""
    return TerminalManager.get_size()


def set_terminal_title(title: str) -> None:
    """Set terminal window title."""
    TerminalManager.set_title(title)


# Additional convenience functions
def hide_cursor() -> None:
    """Hide the terminal cursor."""
    TerminalManager.hide_cursor()


def show_cursor() -> None:
    """Show the terminal cursor."""
    TerminalManager.show_cursor()


def bell() -> None:
    """Sound the terminal bell."""
    TerminalManager.bell()


def hyperlink(url: str, text: Optional[str] = None) -> str:
    """Create a clickable hyperlink."""
    return TerminalManager.hyperlink(url, text)


def save_cursor_position() -> None:
    """Save current cursor position."""
    TerminalManager.save_cursor_position()


def restore_cursor_position() -> None:
    """Restore saved cursor position."""
    TerminalManager.restore_cursor_position()


def move_cursor_up(lines: int = 1) -> None:
    """Move cursor up by specified lines."""
    TerminalManager.move_cursor_up(lines)


def move_cursor_down(lines: int = 1) -> None:
    """Move cursor down by specified lines."""
    TerminalManager.move_cursor_down(lines)


def clear_line() -> None:
    """Clear the current line."""
    TerminalManager.clear_line()


def get_terminal_info() -> dict:
    """Get comprehensive terminal information."""
    return TerminalManager.get_terminal_info()


@contextmanager
def alternate_screen():
    """Context manager for alternate screen buffer."""
    with TerminalManager.alternate_screen():
        yield