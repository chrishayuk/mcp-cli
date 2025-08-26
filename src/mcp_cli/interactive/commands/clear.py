# mcp_cli/interactive/commands/clear.py
"""
Interactive **clear / cls** command for MCP-CLI
===============================================

This module wires the interactive commands **`clear`** and **`cls`**
to the shared :pyfunc:`mcp_cli.commands.clear.clear_action` utility so
users can wipe the screen without touching conversation history.

Why have two names?
-------------------
* ``clear`` - familiar to Unix/Linux and many PowerShell users.
* ``cls``   - classic Windows `cmd.exe` shortcut.

Both aliases call exactly the same function.

Behaviour
---------
* **Visual reset only** - the terminal window is cleared, but all in-memory
  state (conversation, loaded tools, etc.) remains untouched.
* **Cross-platform** - `clear_action()` already detects whether ANSI escape
  codes are supported and falls back gracefully (e.g. on vanilla
  Windows 10 `cmd.exe` or when output is being piped to a file).
* **No arguments** - any extra tokens after the command are ignored.

Examples
--------
>>> clear
>>> cls
"""

from __future__ import annotations

from typing import Any, List

from .base import InteractiveCommand
from mcp_cli.commands.clear import clear_action


class ClearCommand(InteractiveCommand):
    """Erase the visible terminal contents (aliases: *cls*)."""

    def __init__(self) -> None:
        super().__init__(
            name="clear",
            help_text="Clear the terminal screen without affecting session state.",
            aliases=["cls"],
        )

    # ------------------------------------------------------------------
    async def execute(  # noqa: D401 - imperative verb is fine
        self,
        args: List[str],
        tool_manager: Any = None,  # unused but kept for signature parity
        **_: Any,
    ) -> None:
        """
        Ignore *args* and delegate to :pyfunc:`mcp_cli.commands.clear.clear_action`.
        """
        _ = args  # explicitly discard
        clear_action()
