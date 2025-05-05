# src/mcp_cli/commands/tools.py
"""
Shared tools-listing logic for both interactive and CLI interfaces.
"""
import json
from typing import Any, List, Dict
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.formatting import create_tools_table


def tools_action(
    tm: ToolManager,
    *,
    show_details: bool = False,
    show_raw: bool = False
) -> List[Any]:
    """
    Fetch unique tools from the ToolManager and render them.

    If show_raw is True, prints raw JSON; otherwise prints a table.
    Returns the underlying list of ToolInfo or raw dicts.
    """
    console = Console()
    console.print("[cyan]\nFetching Tools List from all servers...[/cyan]")

    all_tools = tm.get_unique_tools()
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
