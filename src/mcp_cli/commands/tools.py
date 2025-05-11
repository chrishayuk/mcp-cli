# src/mcp_cli/commands/tools.py
"""
Shared tools-listing logic for both interactive and CLI interfaces.
"""
import json
import asyncio
from typing import Any, List, Dict
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.formatting import create_tools_table


async def tools_action_async(
    tm: ToolManager,
    *,
    show_details: bool = False,
    show_raw: bool = False
) -> List[Any]:
    """
    Async version: Fetch unique tools from the ToolManager and render them.

    If show_raw is True, prints raw JSON; otherwise prints a table.
    Returns the underlying list of ToolInfo or raw dicts.
    """
    console = Console()
    console.print("[cyan]\nFetching Tools List from all servers...[/cyan]")

    all_tools = await tm.get_unique_tools()
    if not all_tools:
        console.print("[yellow]No tools available from any server.[/yellow]")
        return []

    if show_raw:
        raw_defs: List[Dict[str, Any]] = []
        for t in all_tools:
            raw_defs.append({
                "name": t.name,
                "namespace": t.namespace,
                "description": t.description,
                "parameters": t.parameters,
                "is_async": t.is_async,
                "tags": t.tags,
            })
        text = json.dumps(raw_defs, indent=2)
        console.print(Syntax(text, "json", theme="monokai", line_numbers=True))
        return raw_defs

    # Otherwise show table
    table = create_tools_table(all_tools, show_details=show_details)
    console.print(table)
    console.print(f"[green]Total tools available: {len(all_tools)}[/green]")
    return all_tools


def tools_action(
    tm: ToolManager,
    *,
    show_details: bool = False,
    show_raw: bool = False
) -> List[Any]:
    """
    Synchronous wrapper (legacy): Fetch unique tools from the ToolManager and render them.

    If show_raw is True, prints raw JSON; otherwise prints a table.
    Returns the underlying list of ToolInfo or raw dicts.
    """
    console = Console()
    console.print("[cyan]\nFetching Tools List from all servers...[/cyan]")

    # This is a temporary workaround - use the synchronous version for backward compatibility
    # We'll need to run this in the event loop
    try:
        # Try getting the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we can't use run_until_complete
            console.print("[yellow]Warning: Event loop is running, tool list may be incomplete.[/yellow]")
            all_tools = []
        else:
            # Use the async version if we can
            all_tools = loop.run_until_complete(tm.get_unique_tools())
    except RuntimeError:
        # No event loop - create one
        console.print("[yellow]Warning: Creating new event loop for tool listing.[/yellow]")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        all_tools = loop.run_until_complete(tm.get_unique_tools())
    
    if not all_tools:
        console.print("[yellow]No tools available from any server.[/yellow]")
        return []

    if show_raw:
        raw_defs: List[Dict[str, Any]] = []
        for t in all_tools:
            raw_defs.append({
                "name": t.name,
                "namespace": t.namespace,
                "description": t.description,
                "parameters": t.parameters,
                "is_async": t.is_async,
                "tags": t.tags,
            })
        text = json.dumps(raw_defs, indent=2)
        console.print(Syntax(text, "json", theme="monokai", line_numbers=True))
        return raw_defs

    # Otherwise show table
    table = create_tools_table(all_tools, show_details=show_details)
    console.print(table)
    console.print(f"[green]Total tools available: {len(all_tools)}[/green]")
    return all_tools