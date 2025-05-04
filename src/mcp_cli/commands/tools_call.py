# src/mcp_cli/commands/tools_call.py
from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, Dict
from rich.console import Console

# mcp cli
from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.models import ToolCallResult

# logger
logger = logging.getLogger(__name__)


async def tools_call_action(tm: ToolManager) -> None:
    """
    Prompts the user to select a tool, gather JSON args, 
    execute the tool via ToolManager, and render the result.
    """
    console = Console()
    print = console.print

    print("[cyan]\nTool Call Interface[/cyan]")

    all_tools = tm.get_all_tools()
    if not all_tools:
        print("[yellow]No tools available from any server.[/yellow]")
        return

    # List tools
    print("[green]Available tools:[/green]")
    for idx, tool in enumerate(all_tools, start=1):
        print(f"  {idx}. {tool.name} (from {tool.namespace}) – {tool.description or 'No description'}")

    # Ask for selection
    sel_raw = await asyncio.to_thread(input, "\nEnter tool number to call: ")
    try:
        sel = int(sel_raw) - 1
        if not (0 <= sel < len(all_tools)):
            print("[red]Invalid selection.[/red]")
            return
    except ValueError:
        print("[red]Please enter a valid number.[/red]")
        return

    tool = all_tools[sel]
    print(f"\n[green]Selected:[/green] {tool.name} from {tool.namespace}")
    print(f"[cyan]Description:[/cyan] {tool.description or 'No description'}")

    # Show parameters if any
    params = tool.parameters or {}
    if isinstance(params, dict) and "properties" in params and params["properties"]:
        print("\n[yellow]Parameters:[/yellow]")
        required = set(params.get("required", []))
        for name, details in params["properties"].items():
            req = "[Required]" if name in required else "[Optional]"
            ptype = details.get("type", "any")
            desc = details.get("description", "")
            print(f"  – {name} ({ptype}) {req}: {desc}")

    # Get JSON args
    print("\n[yellow]Enter arguments as JSON (empty for none):[/yellow]")
    args_raw = await asyncio.to_thread(input, "> ")
    if args_raw.strip():
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            print("[red]Invalid JSON. Please try again.[/red]")
            return
    else:
        args = {}

    print(f"\n[cyan]Calling tool '{tool.name}' with arguments:[/cyan]")
    console.print(json.dumps(args, indent=2))

    # Execute
    try:
        result: ToolCallResult = await tm.execute_tool(tool.name, args)
        # Display result nicely
        from mcp_cli.tools.formatting import display_tool_call_result
        display_tool_call_result(result)
    except Exception as e:
        logger.exception("Error executing tool")
        print(f"[red]Error: {e}[/red]")
