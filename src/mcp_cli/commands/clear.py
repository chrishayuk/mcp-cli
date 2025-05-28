# mcp_cli/commands/clear.py
"""
Clear the user's terminal window
================================

This helper is shared by both:

* **Chat-mode** - the `/clear` and `/cls` slash-commands.
* **Non-interactive CLI** - the `mcp-cli clear run` Typer sub-command.

It simply calls :pyfunc:`mcp_cli.ui.ui_helpers.clear_screen` and, if
*verbose* is enabled, prints a tiny confirmation so scripts can detect that
the operation completed.
"""
from __future__ import annotations

# mcp cli
from mcp_cli.ui.ui_helpers import clear_screen
from mcp_cli.utils.rich_helpers import get_console


def clear_action(*, verbose: bool = False) -> None:  # noqa: D401
    """
    Clear whatever terminal the user is running in.

    Parameters
    ----------
    verbose:
        When **True** a dim “Screen cleared.” message is written afterwards
        (useful for log files or when the command is scripted).
    """
    clear_screen()

    if verbose:
        console = get_console()
        console.print("[dim]Screen cleared.[/dim]")
