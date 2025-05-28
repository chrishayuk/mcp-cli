# mcp_cli/utils/rich_helpers.py
from rich.console import Console
import sys, os

def get_console() -> Console:
    """
    Return a Console configured for the current platform / TTY.
    - Disables colour if stdout is redirected.
    - Enables legacy Windows support for very old terminals.
    - Adds soft-wrap to prevent horizontal overflow.
    """
    return Console(
        no_color=not sys.stdout.isatty(),
        legacy_windows=True,     # harmless on mac/Linux, useful on Win ≤8.1
        soft_wrap=True,
    )
