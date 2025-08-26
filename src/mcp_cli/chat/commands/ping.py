# mcp_cli/chat/commands/ping.py
"""
Chat-mode `/ping` command for MCP-CLI
====================================

This module wires the chat-command **/ping** to the shared
:meth:`mcp_cli.commands.ping.ping_action_async` helper so that end-users
can measure the round-trip latency to each MCP server from inside an
interactive chat session.

Key Features
------------
* **Zero state** - the handler is a thin façade; it never mutates the
  context and can be hot-reloaded safely.
* **Filter support** - any additional tokens after */ping* are treated as
  case-insensitive filters that match either the *index* **or** the
  *display-name* of a server (e.g. ``/ping 0 db analytics``).

Usage Examples
--------------
>>> /ping                # ping every connected server
>>> /ping 0 api          # ping only server 0 and the one named "api"

The response is rendered as a Rich table with three columns:
* **Server** - the user-friendly name or index
* **Status** - ✓ on success, ✗ on timeout or error
* **Latency** - round-trip time in milliseconds
"""

from __future__ import annotations

from typing import Any, Dict, List

# Rich console helper (handles Windows quirks, ANSI passthrough, etc.)
from chuk_term.ui import output

# Shared implementation
from mcp_cli.commands.ping import ping_action_async
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command


async def ping_command(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Measure round-trip latency to one or more MCP servers."""

    tm: ToolManager | None = ctx.get("tool_manager")
    if tm is None:
        output.print("[red]Error:[/red] ToolManager not available.")
        return True  # command *was* handled (nothing else to do)

    # Everything after "/ping" is considered a filter (index or name)
    targets = parts[1:]
    return await ping_action_async(tm, targets=targets)


# ---------------------------------------------------------------------------
# Registration (no extra alias - keep namespace clean)
# ---------------------------------------------------------------------------
register_command("/ping", ping_command)
