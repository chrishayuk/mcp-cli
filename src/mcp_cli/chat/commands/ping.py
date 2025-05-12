# mcp_cli/chat/commands/ping.py
"""
Chat-mode `/ping` command – measure latency to connected MCP servers.
"""
from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console
from rich import print

from mcp_cli.commands.ping import ping_action_async      # ← async helper
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command


async def ping_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """
    Usage
    -----
      /ping            Ping all servers
      /ping <filter>   Ping only servers whose index or name matches <filter>
      /p               Alias
    """
    console = Console()
    tm: ToolManager | None = ctx.get("tool_manager")

    if not tm:
        print("[red]Error:[/red] ToolManager not available.")
        return True

    targets = parts[1:]  # anything after /ping is treated as filter text
    success = await ping_action_async(tm, targets=targets)
    return success


# Register command + short alias
register_command("/ping", ping_command)
register_command("/p", ping_command)
