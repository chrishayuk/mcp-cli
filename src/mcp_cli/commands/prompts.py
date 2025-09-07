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

# mcp cli
from mcp_cli.utils.async_utils import run_blocking
from chuk_term.ui import output, format_table
from mcp_cli.context import get_context


# ════════════════════════════════════════════════════════════════════════
# async (primary) implementation
# ════════════════════════════════════════════════════════════════════════
async def prompts_action_async() -> List[Dict[str, Any]]:
    """
    Fetch **all** prompt templates from every connected server and
    display them in a nicely formatted Rich table.

    Returns
    -------
    list[dict]
        The raw prompt dictionaries exactly as returned by `ToolManager`.
    """
    # Get context and tool manager
    context = get_context()
    tm = context.tool_manager

    if not tm:
        output.error("No tool manager available")
        return []

    try:
        maybe = tm.list_prompts()
    except Exception as exc:  # pragma: no cover - network / server errors
        output.error(f"{exc}")
        return []

    # `tm.list_prompts()` can be sync or async - handle both gracefully
    prompts = await maybe if inspect.isawaitable(maybe) else maybe
    if not prompts:  #  None or empty list
        output.info("No prompts recorded.")
        return []

    # Build table data for chuk_term
    table_data = []
    columns = ["Server", "Name", "Description"]

    for item in prompts:
        table_data.append(
            {
                "Server": item.get("server", "-"),
                "Name": item.get("name", "-"),
                "Description": item.get("description", ""),
            }
        )

    # Create and display table using chuk-term
    table = format_table(table_data, title="Prompts", columns=columns)
    output.print_table(table)
    return prompts


# ════════════════════════════════════════════════════════════════════════
# sync wrapper - used by legacy CLI commands
# ════════════════════════════════════════════════════════════════════════
def prompts_action() -> List[Dict[str, Any]]:
    """
    Blocking helper around :pyfunc:`prompts_action_async`.

    It calls :pyfunc:`mcp_cli.utils.async_utils.run_blocking`, raising a
    ``RuntimeError`` if invoked from *inside* a running event-loop.
    """
    return run_blocking(prompts_action_async())


# ════════════════════════════════════════════════════════════════════════
# alias for chat/interactive mode
# ════════════════════════════════════════════════════════════════════════
async def prompts_action_cmd() -> List[Dict[str, Any]]:
    """
    Alias kept for the interactive */prompts* command.

    Chat-mode already runs inside an event-loop, so callers should simply
    `await` this coroutine instead of the synchronous wrapper.
    """
    return await prompts_action_async()


__all__ = [
    "prompts_action_async",
    "prompts_action",
    "prompts_action_cmd",
]
