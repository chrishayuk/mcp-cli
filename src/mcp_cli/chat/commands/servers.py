# mcp_cli/chat/commands/servers.py
"""
List all MCP servers currently connected to this session.

The **/servers** slash-command (alias **/srv**) shows every server that the
active ``ToolManager`` is aware of, together with its friendly name, tool
count and health status.

Features
--------
* **Cross-platform output** – uses
  :pyfunc:`mcp_cli.utils.rich_helpers.get_console` for automatic fallback to
  plain text on Windows consoles or when piping output to a file.
* **Read-only** – purely informational; the command never mutates the chat
  context and is safe to hot-reload.
* **One-liner implementation** – delegates the heavy lifting to the shared
  :pyfunc:`mcp_cli.commands.servers.servers_action_async` helper, ensuring
  a single source of truth between chat and CLI modes.

Examples
--------
  /servers        → tabular list of every connected server  
  /srv            → same, using the shorter alias
"""
from __future__ import annotations

from typing import Any, Dict, List

# Cross-platform Rich console helper
from mcp_cli.utils.rich_helpers import get_console

# Shared async helper
from mcp_cli.commands.servers import servers_action_async
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# Command handler
# ════════════════════════════════════════════════════════════════════════════
async def servers_command(_parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """List all MCP servers currently connected to this session."""
    console = get_console()

    tm: ToolManager | None = ctx.get("tool_manager")
    if tm is None:
        console.print("[red]Error:[/red] ToolManager not available.")
        return True  # command handled

    await servers_action_async(tm)
    return True


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
register_command("/servers", servers_command)
