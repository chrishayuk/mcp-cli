# src/mcp_cli/commands/prompts.py
"""
List stored *prompt* templates on every connected MCP server
============================================================

Public entry-points
-------------------
* **prompts_action_async(tm)** - canonical coroutine (used by chat */prompts*).
* **prompts_action(tm)**       - small synchronous wrapper for plain CLI usage.
* **prompts_action_cmd(tm)**   - thin alias kept for backward-compatibility.

All variants ultimately render the same Rich table:

┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Server ┃ Name       ┃ Description                         ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ local  │ greet      │ Friendly greeting prompt            │
│ api    │ sql_query  │ Extract columns & types from table  │
└────────┴────────────┴─────────────────────────────────────┘
"""

from __future__ import annotations
import inspect
from typing import Any, Dict, List
from rich.table import Table

# mcp cli
from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking
from chuk_term.ui import output


# ════════════════════════════════════════════════════════════════════════
# async (primary) implementation
# ════════════════════════════════════════════════════════════════════════
async def prompts_action_async(tm: ToolManager) -> List[Dict[str, Any]]:
    """
    Fetch **all** prompt templates from every connected server and
    display them in a nicely formatted Rich table.

    Returns
    -------
    list[dict]
        The raw prompt dictionaries exactly as returned by `ToolManager`.
    """

    try:
        maybe = tm.list_prompts()
    except Exception as exc:  # pragma: no cover - network / server errors
        output.print(f"[red]Error:[/red] {exc}")
        return []

    # `tm.list_prompts()` can be sync or async - handle both gracefully
    prompts = await maybe if inspect.isawaitable(maybe) else maybe
    if not prompts:  #  None or empty list
        output.print("[dim]No prompts recorded.[/dim]")
        return []

    # ── render table ────────────────────────────────────────────────────
    table = Table(title="Prompts", header_style="bold magenta")
    table.add_column("Server", style="cyan", no_wrap=True)
    table.add_column("Name", style="yellow", no_wrap=True)
    table.add_column("Description", overflow="fold")

    for item in prompts:
        table.add_row(
            item.get("server", "-"),
            item.get("name", "-"),
            item.get("description", ""),
        )

    output.print(table)
    return prompts


# ════════════════════════════════════════════════════════════════════════
# sync wrapper - used by legacy CLI commands
# ════════════════════════════════════════════════════════════════════════
def prompts_action(tm: ToolManager) -> List[Dict[str, Any]]:
    """
    Blocking helper around :pyfunc:`prompts_action_async`.

    It calls :pyfunc:`mcp_cli.utils.async_utils.run_blocking`, raising a
    ``RuntimeError`` if invoked from *inside* a running event-loop.
    """
    return run_blocking(prompts_action_async(tm))


# ════════════════════════════════════════════════════════════════════════
# alias for chat/interactive mode
# ════════════════════════════════════════════════════════════════════════
async def prompts_action_cmd(tm: ToolManager) -> List[Dict[str, Any]]:
    """
    Alias kept for the interactive */prompts* command.

    Chat-mode already runs inside an event-loop, so callers should simply
    `await` this coroutine instead of the synchronous wrapper.
    """
    return await prompts_action_async(tm)


__all__ = [
    "prompts_action_async",
    "prompts_action",
    "prompts_action_cmd",
]
