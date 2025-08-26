# src/mcp_cli/commands/tools.py
"""
Show **all tools** exposed by every connected MCP server, either as a
pretty Rich table or raw JSON.

How to use
----------
* Chat / interactive : `/tools`, `/tools --all`, `/tools --raw`
* CLI script         : `mcp-cli tools [--all|--raw]`

Both the chat & CLI layers call :pyfunc:`tools_action_async`; the
blocking helper :pyfunc:`tools_action` exists only for legacy sync code.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from rich.syntax import Syntax
from rich.table import Table

# MCP-CLI helpers
from mcp_cli.tools.formatting import create_tools_table
from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking
from chuk_term.ui import output

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────────
# async (canonical) implementation
# ────────────────────────────────────────────────────────────────────────────────
async def tools_action_async(  # noqa: D401
    tm: ToolManager,
    *,
    show_details: bool = False,
    show_raw: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch the **deduplicated** tool list from *all* servers and print it.

    Parameters
    ----------
    tm
        A fully-initialised :class:`~mcp_cli.tools.manager.ToolManager`.
    show_details
        When *True*, include parameter schemas in the table.
    show_raw
        When *True*, dump raw JSON definitions instead of a table.

    Returns
    -------
    list
        The list of tool-metadata dictionaries (always JSON-serialisable).
    """
    output.print("[cyan]\nFetching tool catalogue from all servers…[/cyan]")

    all_tools = await tm.get_unique_tools()
    if not all_tools:
        output.print("[yellow]No tools available from any server.[/yellow]")
        logger.debug("ToolManager returned an empty tools list")
        return []

    # ── raw JSON mode ───────────────────────────────────────────────────
    if show_raw:
        payload = [
            {
                "name": t.name,
                "namespace": t.namespace,
                "description": t.description,
                "parameters": t.parameters,
                "is_async": getattr(t, "is_async", False),
                "tags": getattr(t, "tags", []),
                "aliases": getattr(t, "aliases", []),
            }
            for t in all_tools
        ]
        output.print(
            Syntax(
                json.dumps(payload, indent=2, ensure_ascii=False),
                "json",
                line_numbers=True,
            )
        )
        return payload

    # ── Rich table mode ─────────────────────────────────────────────────
    table: Table = create_tools_table(all_tools, show_details=show_details)
    output.print(table)
    output.print(f"[green]Total tools available: {len(all_tools)}[/green]")

    # Return a safe JSON structure (no .to_dict() needed)
    return [
        {
            "name": t.name,
            "namespace": t.namespace,
            "description": t.description,
            "parameters": t.parameters,
            "is_async": getattr(t, "is_async", False),
            "tags": getattr(t, "tags", []),
            "aliases": getattr(t, "aliases", []),
        }
        for t in all_tools
    ]


# ────────────────────────────────────────────────────────────────────────────────
# sync wrapper - for legacy CLI paths
# ────────────────────────────────────────────────────────────────────────────────
def tools_action(
    tm: ToolManager,
    *,
    show_details: bool = False,
    show_raw: bool = False,
) -> List[Dict[str, Any]]:
    """
    Blocking wrapper around :pyfunc:`tools_action_async`.

    Raises
    ------
    RuntimeError
        If called from inside a running event-loop.
    """
    return run_blocking(
        tools_action_async(tm, show_details=show_details, show_raw=show_raw)
    )


__all__ = ["tools_action_async", "tools_action"]
