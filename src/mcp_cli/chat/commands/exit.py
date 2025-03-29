# src/cli/chat/commands/exit.py
"""
Exit commands for the MCP chat interface.
Includes exit and general utility commands.
"""
from typing import List, Dict, Any
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown

# imports
from mcp_cli.chat.commands import register_command

async def cmd_exit(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Exit the chat session.
    
    Usage: /exit
    """
    # Set exit flag to be caught by the main loop
    context['exit_requested'] = True

    # print
    print(Panel("Exiting chat mode.", style="bold red"))

    # exit
    return True


async def cmd_quit(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Exit the chat session (alias for /exit).
    
    Usage: /quit
    """

    # exit
    return await cmd_exit(cmd_parts, context)


# Register all commands in this module
register_command("/exit", cmd_exit)
register_command("/quit", cmd_quit)