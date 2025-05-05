# mcp_cli/chat/commands/resources.py
"""
Chat command for listing resources from connected MCP servers.
Delegates to the shared resources_action.
"""
from typing import List, Dict, Any
from rich.console import Console

# Shared implementation
from mcp_cli.commands.resources import resources_action
from mcp_cli.tools.manager import ToolManager

# Chat registry
from mcp_cli.chat.commands import register_command

async def cmd_resources(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    List resources recorded by connected servers.

    Usage:
      /resources    — Show all resources
      /res          — Alias for /resources
    """
    console = Console()
    tm: ToolManager = context.get("tool_manager")
    if not tm:
        console.print("[red]Error: no tool manager available[/red]")
        return True

    # resources_action already handles printing
    await resources_action(tm)
    return True

# Register under /resources and /res
register_command("/resources", cmd_resources)
register_command("/res",       cmd_resources)
