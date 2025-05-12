# mcp_cli/commands/clear.py
"""
Cross-platform screen-clear utility for both CLI and interactive modes.
"""
from __future__ import annotations

from rich import print

from mcp_cli.ui.ui_helpers import clear_screen


def clear_action(*, verbose: bool = False) -> None:
    """
    Clear the terminal.

    Parameters
    ----------
    verbose
        When *True*, prints a subtle confirmation after clearing.
    """
    clear_screen()
    if verbose:
        print("[dim]Screen cleared.[/dim]")
