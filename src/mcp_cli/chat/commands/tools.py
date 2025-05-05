# mcp_cli/chat/commands/tools.py
"""
Chat command module for listing and calling available tools,
reusing the shared CLI logic for consistency.
"""
from typing import List, Any
from rich.console import Console

# Shared implementations
from mcp_cli.commands.tools import tools_action
from mcp_cli.commands.tools_call import tools_call_action
from mcp_cli.tools.manager import ToolManager

# Chat registry
from mcp_cli.chat.commands import register_command

async def tools_command(cmd_parts: List[str], context: dict) -> bool:
    """
    Display or call tools in chat mode.

    Usage:
      /tools            List tools
      /tools --all      Show detailed tool info
      /tools --raw      Show raw JSON definitions
      /tools call       Launch interactive tool-call UI
    """
    console = Console()
    tool_manager: ToolManager = context.get("tool_manager")
    if not tool_manager:
        console.print("[red]Error: no tool manager available[/red]")
        return True

    # Parse flags and subcommand
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
    if args and args[0].lower() == "call":
        # Interactive tool call
        await tools_call_action(tool_manager)
    else:
        # Listing mode
        show_all = "--all" in args
        show_raw = "--raw" in args

        # Delegate to shared CLI action
        tools_action(
            tool_manager,
            show_details=show_all,
            show_raw=show_raw
        )

    return True

# Register under /tools and alias /t
register_command("/tools", tools_command, ["--all", "--raw"])
register_command("/t", tools_command, ["--all", "--raw"])