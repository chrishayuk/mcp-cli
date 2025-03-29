# src/cli/chat/commands/help.py
"""
Help commands for the MCP chat interface.
Includes help and general utility commands.
"""
from typing import List, Dict, Any
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.console import Console

# Import the registration function
from mcp_cli.chat.commands import register_command

# Try to import help_text, but don't fail if it's not there yet
try:
    from mcp_cli.chat.commands.help_text import TOOL_COMMANDS_HELP
except ImportError:
    TOOL_COMMANDS_HELP = """
    ## Tool Commands
    
    MCP provides several commands for working with tools:
    
    - `/tools`: List all available tools across connected servers
      - `/tools --all`: Show detailed information including parameters
      - `/tools --raw`: Show raw tool definitions (for debugging)
    
    - `/toolhistory` or `/th`: Show history of tool calls in the current session
      - `/th -n 5`: Show only the last 5 tool calls
      - `/th --json`: Show tool calls in JSON format
    
    - `/interrupt`, `/stop`, or `/cancel`: Interrupt running tool execution
    
    In compact mode (default), tool calls are shown in a condensed format.
    Use `/toolhistory` to see all tools that have been called in the session.
    """

async def cmd_help(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display help information for all commands or a specific command.
    
    Usage: 
      /help           - Show all commands
      /help <command> - Show help for a specific command
      /help tools     - Show help about tool-related commands
      /help conversation - Show help about conversation history commands
    """
    console = Console()
    
    # Special case for tool commands help
    if len(cmd_parts) > 1 and cmd_parts[1].lower() == "tools":
        print(Panel(Markdown(TOOL_COMMANDS_HELP), style="cyan", title="Tool Commands Help"))
        return True
    
    # Special case for conversation history help
    if len(cmd_parts) > 1 and cmd_parts[1].lower() in ("conversation", "ch"):
        from mcp_cli.chat.commands.conversation_history import conversation_history_command
        help_text = conversation_history_command.__doc__ or "No detailed help available for conversation history."
        print(Panel(Markdown(help_text), style="cyan", title="Conversation History Help"))
        return True

    # Help for a specific command
    if len(cmd_parts) > 1:
        specific_cmd = cmd_parts[1].lower()
        if not specific_cmd.startswith('/'):
            specific_cmd = '/' + specific_cmd
            
        # Use the show_command_help functionality from the module
        from mcp_cli.chat.commands import _COMMAND_HANDLERS
        
        if specific_cmd in _COMMAND_HANDLERS:
            handler = _COMMAND_HANDLERS[specific_cmd]
            
            # Extract module name for the heading
            module_name = handler.__module__.split('.')[-1].capitalize()
            
            help_text = f"## {specific_cmd}\n\n"
            
            if handler.__doc__:
                help_text += handler.__doc__.strip()
            else:
                help_text += "No detailed help available for this command."
            
            # Add completions info if available
            from mcp_cli.chat.commands import _COMMAND_COMPLETIONS
            if specific_cmd in _COMMAND_COMPLETIONS:
                help_text += "\n\n### Completions\n\n"
                help_text += ", ".join(_COMMAND_COMPLETIONS[specific_cmd])
            
            print(Panel(Markdown(help_text), style="cyan", title=f"Help: {specific_cmd} ({module_name})"))
        else:
            print(f"[yellow]Command {specific_cmd} not found. Try /help for a list of all commands.[/yellow]")
        
        return True
    
    # General help - get all commands
    from mcp_cli.chat.commands import _COMMAND_HANDLERS
    
    # List of commands to exclude
    excluded_commands = ["/verbose", "/v"]
    
    # Filter out excluded commands
    visible_commands = {cmd: handler for cmd, handler in _COMMAND_HANDLERS.items() 
                      if cmd not in excluded_commands}
    
    # Count total commands
    total_commands = len(visible_commands)
    
    # Create a table like the tools command
    table = Table(title=f"{total_commands} Available Commands")
    
    # Add columns - removed the Category column
    table.add_column("Command", style="green")
    table.add_column("Description")
    
    # Sort all commands
    sorted_commands = sorted(visible_commands.items())
    
    # Add rows for each command
    for cmd, handler in sorted_commands:
        # Extract description from docstring
        desc = "No description"
        if handler.__doc__:
            # Get the first non-empty line from the docstring
            for line in handler.__doc__.strip().split('\n'):
                line = line.strip()
                if line:
                    desc = line
                    break
        
        # Truncate long descriptions
        if len(desc) > 75:
            desc = desc[:72] + "..."
            
        table.add_row(cmd, desc)
    
    # Print the table
    console.print(table)
    
    # Show help note
    console.print("\nType [green]/help <command>[/green] for more information about a specific command.")
    console.print("For detailed information about tool commands, type [green]/help tools[/green].")
    console.print("For conversation history, type [green]/help conversation[/green] or [green]/help ch[/green].")
    
    return True


async def display_quick_help(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display a quick reference of common commands.
    
    Usage: /quickhelp
    """
    console = Console()
    
    # Create a table for common commands - same style as tools
    table = Table(title="Quick Command Reference")
    
    # Add columns - removed Category column
    table.add_column("Command", style="green")
    table.add_column("Description")
    
    # Add rows for common commands - now including conversation history
    table.add_row("/help", "Display detailed help")
    table.add_row("/tools", "List available tools")
    table.add_row("/toolhistory, /th", "View history of tool calls")
    table.add_row("/conversation, /ch", "Show conversation history")
    table.add_row("/clear", "Clear conversation or screen")
    table.add_row("/interrupt, /stop", "Interrupt running tools")
    table.add_row("exit, quit", "Exit chat mode")
    
    console.print(table)
    console.print("\nType [green]/help[/green] for complete command listing or [green]/help <command>[/green] for details.")
    
    return True

# Register all commands in this module
register_command("/help", cmd_help)
register_command("/quickhelp", display_quick_help)
register_command("/qh", display_quick_help)