# src/mcp_cli/commands/actions/exit.py
"""
Exit action for MCP CLI.

Provides functionality to cleanly terminate the MCP CLI session.

Public functions:
* **exit_action()** - Exit the application with cleanup.
"""

from __future__ import annotations

import sys
from chuk_term.ui import output, restore_terminal


def exit_action(interactive: bool = True) -> bool:
    """
    Cleanly exit the MCP CLI session.

    Args:
        interactive: If True, return to allow outer loop to break.
                    If False, call sys.exit(0) to terminate process.

    Returns:
        True when interactive mode (to signal loop break).
        Never returns when non-interactive (process exits).
    """
    output.info("Exiting… Goodbye!")
    restore_terminal()

    if not interactive:
        sys.exit(0)

    return True


__all__ = ["exit_action"]
