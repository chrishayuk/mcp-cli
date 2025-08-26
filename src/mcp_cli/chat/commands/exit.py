# mcp_cli/chat/commands/exit.py
"""
Chat-mode “/exit” and “/quit” commands for MCP-CLI
==================================================

Both commands perform a single task: **politely end the current chat
session**.

* They set ``context["exit_requested"] = True`` - the main chat loop checks
  this flag and breaks.
* A red confirmation panel is printed so the user knows the request was
  acknowledged.
* No other session state is mutated, making the handler safe to hot-reload.

The module uses :pyfunc:`mcp_cli.utils.rich_helpers.get_console`, which
automatically falls back to plain text when ANSI colours are unavailable
(e.g. legacy Windows consoles or when piping output to a file).
"""

from __future__ import annotations

from typing import Any, Dict, List

# Cross-platform Rich console helper
from chuk_term.ui import output

# Chat-command registry
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# Core handlers
# ════════════════════════════════════════════════════════════════════════════
async def cmd_exit(_parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    Terminate the chat session.

    Usage
    -----
      /exit
    """
    ctx["exit_requested"] = True
    output.panel("Exiting chat mode.", style="red", title="Goodbye")
    return True


async def cmd_quit(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    Terminate the chat session.

    Usage
    -----
      /quit
    """
    return await cmd_exit(parts, ctx)


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
register_command("/exit", cmd_exit)
register_command("/quit", cmd_quit)
