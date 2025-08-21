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