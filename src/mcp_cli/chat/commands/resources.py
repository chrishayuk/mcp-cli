# mcp_cli/chat/commands/resources.py
"""
Chat-mode `/resources` command – list resources discovered by MCP servers.
"""
from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console
from rich import print

from mcp_cli.commands.resources import resources_action_async  # ← async helper
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command


async def cmd_resources(_parts: List[str], ctx: Dict[str, Any]) -> bool:
    """
    Usage
    -----
      /resources   Show resources
      /res         Alias
    """
    console = Console()
    tm: ToolManager | None = ctx.get("tool_manager")

    if not tm:
        print("[red]Error:[/red] ToolManager not available.")
        return True

    await resources_action_async(tm)
    return True


# Register command & alias
register_command("/resources", cmd_resources)
register_command("/res", cmd_resources)
