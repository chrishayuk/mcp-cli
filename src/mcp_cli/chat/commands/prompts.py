# mcp_cli/chat/commands/prompts.py
"""
Chat command for listing prompts from connected MCP servers.
Delegates to the shared prompts_action.
"""

from typing import List, Dict, Any
from rich.console import Console

# Shared implementation
from mcp_cli.commands.prompts import prompts_action
from mcp_cli.tools.manager import ToolManager

# Chat registry
from mcp_cli.chat.commands import register_command

async def cmd_prompts(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    List prompts recorded by connected servers.

    Usage:
      /prompts    — Show all prompts
      /p          — Alias for /prompts
    """
    console = Console()
    tm: ToolManager = context.get("tool_manager")
    if not tm:
        console.print("[red]Error: no tool manager available[/red]")
        return True

    # Delegate to the shared action
    await prompts_action(tm)
    return True

# Register under /prompts and /p
register_command("/prompts", cmd_prompts)
register_command("/p",        cmd_prompts)
