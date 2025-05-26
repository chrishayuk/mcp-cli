# mcp_cli/commands/exit.py
"""
Terminate the current MCP-CLI session
=====================================

Used by both chat-mode (/exit | /quit) **and** the non-interactive CLI's
`exit` sub-command.  It restores the TTY first, then either returns to the
caller (interactive) or exits the process (one-shot mode).
"""
from __future__ import annotations
import sys

# mcp cli
from mcp_cli.ui.ui_helpers import restore_terminal
from mcp_cli.utils.rich_helpers import get_console


def exit_action(interactive: bool = True) -> bool:  # noqa: D401
    """
    Cleanly close the current MCP-CLI session.

    Parameters
    ----------
    interactive
        • **True**  → just tell the outer loop to break and *return*  
        • **False** → restore the TTY **then** call :pyfunc:`sys.exit(0)`

    Returns
    -------
    bool
        Always ``True`` so interactive callers can treat it as a
        “please-stop” flag.  (When *interactive* is ``False`` the function
        never returns because the process terminates.)
    """
    console = get_console()          # unified Rich/plain console
    console.print("[yellow]Exiting… Goodbye![/yellow]")

    restore_terminal()

    if not interactive:
        sys.exit(0)

    return True

