# mcp_cli/chat/commands/servers.py
"""
Chat-mode `/servers` command – lists all connected MCP servers.
"""
from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console
from rich import print

from mcp_cli.commands.servers import servers_action_async  # ← async helper
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command


async def servers_command(_parts: List[str], ctx: Dict[str, Any]) -> bool:
    """
    Usage
    -----
      /servers     List servers
      /srv         Alias
    """
    console = Console()
    tm: ToolManager | None = ctx.get("tool_manager")

    if not tm:
        print("[red]Error:[/red] ToolManager not available.")
        return True

    await servers_action_async(tm)
    return True


# Register command + short alias
register_command("/servers", servers_command)
register_command("/srv", servers_command)
