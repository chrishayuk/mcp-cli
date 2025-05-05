# mcp_cli/chat/commands/servers.py
"""
Chat command module for listing connected MCP servers,
reusing the shared CLI logic for consistency.
"""
from typing import List, Any, Dict
from rich.console import Console

# Shared implementation
from mcp_cli.commands.servers import servers_action
from mcp_cli.tools.manager import ToolManager

# Chat registration helper
from mcp_cli.chat.commands import register_command

async def servers_command(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display a table of all connected servers and their status.

    Usage:
      /servers       - List all servers
      /srv           - Alias for /servers
    """
    console = Console()

    tm: ToolManager = context.get("tool_manager")
    if not tm:
        console.print("[red]Error: no tool manager available[/red]")
        return True

    # Delegate to shared CLI action
    servers_action(tm)
    return True

# Register under /servers and /srv
register_command("/servers", servers_command)
register_command("/srv",     servers_command)
