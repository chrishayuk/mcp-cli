# mcp_cli/chat/commands/help.py
"""
Chat-mode help commands for MCP CLI.

Displays help for all registered chat commands, or detailed help for a specific command.
"""
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# Chat registry
from mcp_cli.chat.commands import register_command, _COMMAND_HANDLERS, _COMMAND_COMPLETIONS

# Optional grouped help text
from mcp_cli.chat.commands.help_text import TOOL_COMMANDS_HELP, CONVERSATION_COMMANDS_HELP

async def cmd_help(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Show help for chat commands.

    Usage:
      /help                — List all commands
      /help <command>      — Show detailed help for one command
      /help tools          — Show grouped tool commands help
      /help conversation   — Show grouped conversation history help
    """
    console = Console()
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []

    # Grouped help
    if args and args[0].lower() in ("tools",):
        console.print(Panel(Markdown(TOOL_COMMANDS_HELP), title="Tool Commands Help", style="cyan"))
        return True
    if args and args[0].lower() in ("conversation", "ch"):
        console.print(Panel(Markdown(CONVERSATION_COMMANDS_HELP), title="Conversation Commands Help", style="cyan"))
        return True

    # Specific command help
    if args and args[0].startswith("/"):
        name = args[0]
    elif args:
        name = "/" + args[0]
    else:
        name = None

    if name and name in _COMMAND_HANDLERS:
        handler = _COMMAND_HANDLERS[name]
        title = f"Help: {name}"
        doc = handler.__doc__ or "No detailed help available."
        text = f"## {name}\n\n{doc.strip()}"
        # Append completions if any
        if name in _COMMAND_COMPLETIONS:
            comps = ", ".join(_COMMAND_COMPLETIONS[name])
            text += f"\n\n**Completions:** {comps}"
        console.print(Panel(Markdown(text), title=title, style="cyan"))
        return True

    # Fallback to listing all
    visible = sorted(_COMMAND_HANDLERS.items())
    table = Table(title=f"{len(visible)} Available Commands")
    table.add_column("Command", style="green")
    table.add_column("Description")

    for cmd, handler in visible:
        # Skip internal or verbose-only commands if desired
        desc = "No description"
        if handler.__doc__:
            # first non-empty line
            for line in handler.__doc__.splitlines():
                if line.strip():
                    desc = line.strip()
                    break
        table.add_row(cmd, desc)

    console.print(table)
    console.print("\nType [green]/help <command>[/green] for more details.")
    return True

async def display_quick_help(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Show a quick reference of the most common commands.

    Usage: /quickhelp
    """
    console = Console()
    table = Table(title="Quick Command Reference")
    table.add_column("Command", style="green")
    table.add_column("Description")

    entries = [
        ("/help",       "Show this help"),
        ("/tools",      "List available tools"),
        ("/toolhistory","Show history of tool calls"),
        ("/conversation","Show conversation history"),
        ("/clear",      "Clear screen/history"),
        ("/interrupt",  "Interrupt running tools"),
        ("/exit",       "Exit chat mode"),
    ]

    for cmd, desc in entries:
        table.add_row(cmd, desc)

    console.print(table)
    console.print("\nType [green]/help[/green] for full command list.")
    return True

# Register commands and aliases
register_command("/help",      cmd_help)
register_command("/quickhelp", display_quick_help)
register_command("/qh",        display_quick_help)
