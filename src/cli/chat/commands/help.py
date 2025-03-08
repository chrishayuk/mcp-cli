# src/cli/chat/commands/help.py
"""
Help commands for the MCP chat interface.
Includes help and general utility commands.
"""
from typing import List, Dict, Any
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown

# imports
from cli.chat.commands import register_command


async def cmd_help(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display help information for all commands or a specific command.
    
    Usage: /help [command]
    """
    from cli.commands import get_help_text
    
    if len(cmd_parts) > 1:
        # Help for a specific command
        specific_cmd = cmd_parts[1].lower()
        if not specific_cmd.startswith('/'):
            specific_cmd = '/' + specific_cmd
            
        # Get all commands
        from cli.commands import _COMMAND_HANDLERS
        if specific_cmd in _COMMAND_HANDLERS:
            handler = _COMMAND_HANDLERS[specific_cmd]
            help_text = f"## {specific_cmd}\n\n"
            if handler.__doc__:
                help_text += handler.__doc__.strip()
            else:
                help_text += "No detailed help available for this command."
                
            print(Panel(Markdown(help_text), style="cyan", title=f"Help: {specific_cmd}"))
        else:
            print(f"[yellow]Command {specific_cmd} not found. Try /help for a list of all commands.[/yellow]")
    else:
        # General help
        help_text = get_help_text()
        print(Panel(Markdown(help_text), style="cyan", title="Help"))
        
    return True

# Register all commands in this module
register_command("/help", cmd_help)