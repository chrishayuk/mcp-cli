# src/mcp_cli/commands/tools_call.py
"""
Open an *interactive “call a tool” wizard* that lets you pick a tool and
pass JSON arguments right from the terminal.

Highlights
----------
* Leaves **zero state** behind - safe to hot-reload while a chat/TUI is
  running.
* Re-uses :pyfunc:`mcp_cli.tools.formatting.display_tool_call_result`
  for pretty result rendering, so the output looks the same everywhere.
"""

from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, Dict

# mcp cli
from chuk_term.ui import output
from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.models import ToolCallResult
from mcp_cli.tools.formatting import display_tool_call_result

# logger
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Main entry-point (async coroutine)
# ════════════════════════════════════════════════════════════════════════
async def tools_call_action(tm: ToolManager) -> None:  # noqa: D401
    """
    Launch the mini-wizard, execute the chosen tool, show the result.

    This function is designed for *interactive* use only - it blocks on
    `input()` twice (tool selection & JSON args).
    """
    cprint = output.print

    cprint("[cyan]\nTool Call Interface[/cyan]")

    # Fetch distinct tools (no duplicates across servers)
    all_tools = await tm.get_unique_tools()
    if not all_tools:
        cprint("[yellow]No tools available from any server.[/yellow]")
        return

    # ── list tools ────────────────────────────────────────────────────
    cprint("[green]Available tools:[/green]")
    for idx, tool in enumerate(all_tools, 1):
        desc = tool.description or "No description"
        cprint(f"  {idx}. {tool.name} (from {tool.namespace}) - {desc}")

    # ── user selection ────────────────────────────────────────────────
    sel_raw = await asyncio.to_thread(input, "\nEnter tool number to call: ")
    try:
        sel = int(sel_raw) - 1
        tool = all_tools[sel]
    except (ValueError, IndexError):
        cprint("[red]Invalid selection.[/red]")
        return

    cprint(f"\n[green]Selected:[/green] {tool.name} from {tool.namespace}")
    if tool.description:
        cprint(f"[cyan]Description:[/cyan] {tool.description}")

    # ── argument collection ───────────────────────────────────────────
    params_schema: Dict[str, Any] = tool.parameters or {}
    args: Dict[str, Any] = {}

    if params_schema.get("properties"):
        cprint("\n[yellow]Enter arguments as JSON (leave blank for none):[/yellow]")
        args_raw = await asyncio.to_thread(input, "> ")
        if args_raw.strip():
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                cprint("[red]Invalid JSON - aborting.[/red]")
                return
    else:
        cprint("[dim]Tool takes no arguments.[/dim]")

    # ── execution ─────────────────────────────────────────────────────
    fq_name = f"{tool.namespace}.{tool.name}"
    cprint(f"\n[cyan]Calling '{fq_name}'…[/cyan]")

    try:
        result: ToolCallResult = await tm.execute_tool(fq_name, args)
        display_tool_call_result(None, result, console)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error executing tool")
        cprint(f"[red]Error: {exc}[/red]")


__all__ = ["tools_call_action"]
