# mcp_cli/interactive/commands/exit.py
"""
Interactive **/exit** command for MCP-CLI
=========================================

This tiny wrapper wires the interactive _`exit`_ / _`quit`_ / _`q`_
commands to the shared :pyfunc:`mcp_cli.commands.exit.exit_action`
helper so users can leave the chat shell cleanly.

Key points
----------
* **Graceful shutdown** - delegates all teardown (terminal restore,
  asyncio cleanup, etc.) to *exit_action*.
* **Multiple aliases** - `exit`, `quit`, and `q` all resolve to the same
  behaviour so muscle-memory from other shells still works.
* **Stateless** - no mutation of the chat context; the helper returns
  **True** which the interactive loop interprets as “stop”.
* **Cross-platform** - prints via *exit_action*, which already uses the
  Rich console helper that falls back to plain text on Windows /
  non-ANSI environments.

Examples
--------
>>> exit
>>> quit
>>> q
"""

from __future__ import annotations

from typing import Any, List

from .base import InteractiveCommand
from mcp_cli.commands.exit import exit_action


class ExitCommand(InteractiveCommand):
    """Terminate the interactive chat session (aliases: *quit*, *q*)."""

    def __init__(self) -> None:
        super().__init__(
            name="exit",
            help_text="Quit the interactive shell (aliases: quit, q).",
            aliases=["quit", "q"],
        )

    # ------------------------------------------------------------------
    async def execute(  # noqa: D401 - imperative verb is fine here
        self,
        args: List[str],
        tool_manager: Any = None,  # unused
        **_: Any,
    ) -> bool:
        """
        Invoke :pyfunc:`mcp_cli.commands.exit.exit_action`.

        *Any* trailing arguments are ignored - the command is always
        executed immediately.
        """
        _ = args  # noqa: F841 - explicitly ignore
        return exit_action()  # prints goodbye + returns True
