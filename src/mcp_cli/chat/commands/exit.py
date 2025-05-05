# mcp_cli/chat/commands/exit.py
"""
Chat‐mode exit commands for MCP CLI.
Provides `/exit` and `/quit` to terminate the session gracefully.
"""
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel

# Chat registry
from mcp_cli.chat.commands import register_command

async def cmd_exit(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Exit the chat session.

    Usage: /exit
    """
    console = Console()
    # Signal the main loop to stop
    context["exit_requested"] = True
    console.print(Panel("Exiting chat mode.", style="bold red"))
    return True

async def cmd_quit(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Exit the chat session (alias for /exit).

    Usage: /quit
    """
    return await cmd_exit(cmd_parts, context)

# Register commands
register_command("/exit", cmd_exit)
register_command("/quit", cmd_quit)
