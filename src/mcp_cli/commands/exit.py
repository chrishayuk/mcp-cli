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

# NEW: Import from the new UI module
from chuk_term.ui import (
    output,
    restore_terminal,
)


def exit_action(interactive: bool = True) -> bool:
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
        "please-stop" flag.  (When *interactive* is ``False`` the function
        never returns because the process terminates.)
    """
    # Use the UI output for the goodbye message
    output.info("Exiting… Goodbye!")

    # restore the terminal
    restore_terminal()

    # exit
    if not interactive:
        sys.exit(0)

    # exited
    return True
