# src/mcp_cli/commands/tools.py
"""
Shared tools-listing logic for both interactive and CLI interfaces.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from mcp_cli.tools.formatting import create_tools_table
from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking


# ──────────────────────────────────────────────────────────────────
# async (primary) implementation
# ──────────────────────────────────────────────────────────────────
async def tools_action_async(
    tm: ToolManager,
    *,
    show_details: bool = False,
    show_raw: bool = False,
) -> List[Any]:
    """
    Fetch unique tools and render either a table or raw JSON.

    Returns the list for callers/tests.
    """
    console = Console()
    console.print("[cyan]\nFetching Tools list from all servers…[/cyan]")

    all_tools = await tm.get_unique_tools()
    if not all_tools:
        console.print("[yellow]No tools available from any server.[/yellow]")
        return []

    if show_raw:
        raw_defs: List[Dict[str, Any]] = [
            {
                "name": t.name,
                "namespace": t.namespace,
                "description": t.description,
                "parameters": t.parameters,
                "is_async": t.is_async,
                "tags": t.tags,
            }
            for t in all_tools
        ]
        console.print(Syntax(json.dumps(raw_defs, indent=2), "json", line_numbers=True))
        return raw_defs

    table: Table = create_tools_table(all_tools, show_details=show_details)
    console.print(table)
    console.print(f"[green]Total tools available: {len(all_tools)}[/green]")
    return all_tools


# ──────────────────────────────────────────────────────────────────
# sync wrapper – raises if called from an already-running loop
# ──────────────────────────────────────────────────────────────────
def tools_action(
    tm: ToolManager,
    *,
    show_details: bool = False,
    show_raw: bool = False,
) -> List[Any]:
    return run_blocking(
        tools_action_async(tm, show_details=show_details, show_raw=show_raw)
    )
