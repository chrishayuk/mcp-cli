# src/mcp_cli/commands/tools_call.py
"""
Interactive “call a tool with JSON args” utility.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from rich.console import Console

from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.models import ToolCallResult
from mcp_cli.tools.formatting import display_tool_call_result

logger = logging.getLogger(__name__)


async def tools_call_action(tm: ToolManager) -> None:
    console = Console()
    print_rich = console.print

    print_rich("[cyan]\nTool Call Interface[/cyan]")

    # unique tools to avoid dupes
    all_tools = await tm.get_unique_tools()
    if not all_tools:
        print_rich("[yellow]No tools available from any server.[/yellow]")
        return

    # list tools
    print_rich("[green]Available tools:[/green]")
    for idx, tool in enumerate(all_tools, 1):
        print_rich(
            f"  {idx}. {tool.name} (from {tool.namespace}) – "
            f"{tool.description or 'No description'}"
        )

    # user selection
    sel_raw = await asyncio.to_thread(input, "\nEnter tool number to call: ")
    try:
        sel = int(sel_raw) - 1
        tool = all_tools[sel]
    except (ValueError, IndexError):
        print_rich("[red]Invalid selection.[/red]")
        return

    print_rich(f"\n[green]Selected:[/green] {tool.name} from {tool.namespace}")
    if tool.description:
        print_rich(f"[cyan]Description:[/cyan] {tool.description}")

    # arguments?
    params_schema: Dict[str, Any] = tool.parameters or {}
    args: Dict[str, Any] = {}

    if params_schema.get("properties"):
        print_rich("\n[yellow]Enter arguments as JSON (empty for none):[/yellow]")
        args_raw = await asyncio.to_thread(input, "> ")
        if args_raw.strip():
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                print_rich("[red]Invalid JSON – aborting.[/red]")
                return
    else:
        print_rich("[dim]Tool takes no arguments.[/dim]")

    # execute
    fq_name = f"{tool.namespace}.{tool.name}"
    print_rich(f"\n[cyan]Calling '{fq_name}'…[/cyan]")
    try:
        result: ToolCallResult = await tm.execute_tool(fq_name, args)
        display_tool_call_result(result, console)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error executing tool")
        print_rich(f"[red]Error: {exc}[/red]")
