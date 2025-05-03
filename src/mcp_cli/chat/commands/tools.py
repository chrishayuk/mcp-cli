# mcp_cli/chat/commands/tools.py
"""
Tools command module for listing available tools with their server sources.
"""
from rich.console import Console
from rich.syntax import Syntax
import json

# Import the registration function from the command handler
from mcp_cli.chat.commands import register_command

# Import our formatting helpers
from mcp_cli.tools.formatting import create_tools_table

async def tools_command(cmd_parts, context):
    """Display all available tools with their server information."""
    console = Console()
    
    # Parse arguments
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
    show_all = "--all" in args
    show_raw = "--raw" in args
    
    # Get the tool manager from context
    tool_manager = context.get("tool_manager")
    if not tool_manager:
        console.print("[red]Error: Tool manager not available.[/red]")
        return True
    
    # Get unique tools from the tool manager
    tool_infos = tool_manager.get_unique_tools()
    
    if not tool_infos:
        console.print("[yellow]No tools available.[/yellow]")
        return True
    
    # Show raw JSON if requested
    if show_raw:
        tools_raw = []
        for tool in tool_infos:
            tools_raw.append({
                'name': tool.name,
                'namespace': tool.namespace,
                'description': tool.description,
                'parameters': tool.parameters,
                'is_async': tool.is_async,
                'tags': tool.tags
            })
        raw_json = json.dumps(tools_raw, indent=2)
        console.print(Syntax(raw_json, "json", theme="monokai", line_numbers=True))
        return True
    
    # Use the centralized formatting helper to create the table
    table = create_tools_table(tool_infos, show_details=show_all)
    console.print(table)
    
    if show_all:
        console.print("[yellow]* Required parameter[/yellow]")
    
    return True

# Register the command with its options
register_command("/tools", tools_command, ["--all", "--raw"])