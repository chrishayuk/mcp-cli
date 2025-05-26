# mcp_cli/interactive/commands/help.py
"""
Interactive **help** command - show a list of all commands or drill-down
into a single command's documentation.

Usage examples
--------------
  help            → table of every command with a one-liner description
  help tools      → detailed help for the *tools* command
  h provider      → same, using short alias
  ?               → same as plain *help*

Why this file exists
--------------------
The CLI layer already has `mcp_cli.commands.help.help_action`.  
This interactive wrapper merely:

1.  Grabs a cross-platform Rich console via
    :pyfunc:`mcp_cli.utils.rich_helpers.get_console` so colours work on
    Windows + piped output.
2.  Passes the optional `<command>` argument straight through to
    :func:`help_action`.
"""

from __future__ import annotations

from typing import Any, List, Optional

from mcp_cli.utils.rich_helpers import get_console
from mcp_cli.commands.help import help_action
from .base import InteractiveCommand


class HelpCommand(InteractiveCommand):
    """Display all commands or detailed help for one command."""

    def __init__(self) -> None:
        super().__init__(
            name="help",
            aliases=["h", "?"],  # keep compatibility with old aliases
            help_text="Show global help or detailed help for a specific command.",
        )

    # ------------------------------------------------------------------
    async def execute(  # noqa: D401
        self,
        args: List[str],
        tool_manager: Any = None,  # unused but kept for interface parity
        **_: Any,
    ) -> None:
        """
        Relay to :func:`mcp_cli.commands.help.help_action`.

        *args* is everything after the command word.
        """
        console = get_console()

        # First positional token (if any) is treated as command name.
        # Strip a leading “/” so users can type either form.
        cmd_name: Optional[str] = args[0].lstrip("/") if args else None

        # help_action is synchronous
        help_action(cmd_name, console=console)
