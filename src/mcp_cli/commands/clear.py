# mcp_cli/commands/clear.py
"""
Clear the user's terminal window
================================

This helper is shared by both:

* **Chat-mode** - the `/clear` and `/cls` slash-commands.
* **Non-interactive CLI** - the `mcp-cli clear run` Typer sub-command.

It simply calls :pyfunc:`mcp_cli.ui.clear_screen` and, if
*verbose* is enabled, prints a tiny confirmation so scripts can detect that
the operation completed.
"""

from __future__ import annotations

# NEW: Import from the new UI module
from chuk_term.ui import (
    output,
    clear_screen,
)


def clear_action(*, verbose: bool = False) -> None:
    """
    Clear whatever terminal the user is running in.

    Parameters
    ----------
    verbose:
        When **True** a dim "Screen cleared." message is written afterwards
        (useful for log files or when the command is scripted).
    """
    clear_screen()

    if verbose:
        # NEW: Use the UI output with hint style for subtle message
        output.hint("Screen cleared.")
