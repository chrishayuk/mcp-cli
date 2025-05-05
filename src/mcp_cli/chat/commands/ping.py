# mcp_cli/chat/commands/ping.py
"""
Chat command module for pinging connected MCP servers,
reusing the shared CLI logic for consistency.
"""
from typing import List, Any, Dict
from rich.console import Console
from mcp_cli.commands.ping import ping_action
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command

async def ping_command(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Ping connected servers (optionally filter by index or name).

    Usage:
      /ping             - Ping all servers
      /ping <target>    - Ping only servers matching index or name
      /p                - Alias for /ping
    """
    console = Console()
    tm: ToolManager = context.get("tool_manager")
    if not tm:
        console.print("[red]Error: no tool manager available[/red]")
        return True

    # Skip the "/ping" itself; anything else is a filter target
    targets = cmd_parts[1:]
    success = await ping_action(tm, None, targets)
    # ping_action already prints its table or error
    return success

# Register under /ping and /p
register_command("/ping", ping_command)
register_command("/p",     ping_command)
