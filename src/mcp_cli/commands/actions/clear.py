# src/mcp_cli/commands/actions/clear.py
"""
Clear action for MCP CLI.

Provides functionality to clear the terminal screen.

Public functions:
* **clear_action()** - Clear the terminal screen with optional verbose output.
"""

from __future__ import annotations

from chuk_term.ui import output, clear_screen


def clear_action(*, verbose: bool = False) -> None:
    """
    Clear the terminal screen.

    Args:
        verbose: If True, print a confirmation message after clearing.
    """
    clear_screen()

    if verbose:
        output.hint("Screen cleared.")


__all__ = ["clear_action"]
