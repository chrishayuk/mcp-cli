# mcp_cli/commands/exit.py
"""
Shared exit / quit helper used by both interactive and CLI UIs.
"""
from __future__ import annotations

import sys
from rich import print

from mcp_cli.ui.ui_helpers import restore_terminal


def exit_action(interactive: bool = True) -> bool:
    """
    Cleanly terminate the session.

    Parameters
    ----------
    interactive
        • **True**  → signal the interactive loop to break (returns ``True``).  
        • **False** → exit the whole process via ``sys.exit(0)``.

    Returns
    -------
    bool
        Always ``True`` so the interactive driver knows to stop.
    """
    print("[yellow]Exiting… Goodbye![/yellow]")
    restore_terminal()

    if not interactive:
        sys.exit(0)

    return True
