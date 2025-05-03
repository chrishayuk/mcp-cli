# mcp_cli/commands/prompts.py
"""
Show all prompts advertised by every connected MCP server.

The function can be called either synchronously **or** asynchronously:

* In interactive mode the caller is already inside an event-loop and
  will `await` the coroutine we return (if any).
* From a normal CLI command no loop is running, so we run the coroutine
  to completion ourselves and then render the table directly.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from mcp_cli.tools.manager import ToolManager, get_tool_manager

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _render(prompts: List[Dict[str, Any]], console: Console) -> None:
    """Pretty-print the prompt list as a Rich table."""
    if not prompts:
        console.print("[dim]No prompts recorded.[/dim]")
        return

    table = Table(title="Prompts", header_style="bold magenta")
    table.add_column("Server", style="cyan")
    table.add_column("Name", style="yellow")
    table.add_column("Description", overflow="fold")

    for item in prompts:
        table.add_row(
            item.get("server", "-"),
            item.get("name", "-"),
            item.get("description", ""),
        )

    console.print(table)


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
def prompts_list(
    *,
    stream_manager: Optional[ToolManager] = None,
    console: Optional[Console] = None,
) -> List[Dict[str, Any]] | asyncio.Future:
    """
    List every prompt from every server.

    Returns the raw list so callers/tests can inspect it.
    If the underlying call is asynchronous we **return the awaitable**
    instead – the interactive shell will `await` it, non-async callers
    fall back to running it to completion immediately.
    """
    console = console or Console()
    tm = stream_manager or get_tool_manager()
    if tm is None:
        console.print("[red]Error:[/red] no ToolManager available")
        return []

    result = tm.list_prompts()  # may be a list or a coroutine
    if inspect.isawaitable(result):
        # -----------------------------------------------------------------
        # 1️⃣ interactive mode will await this; non-interactive callers
        #    can decide whether they want to await or not.
        # -----------------------------------------------------------------
        async def _runner() -> List[Dict[str, Any]]:  # inner helper
            data = await result  # type: ignore[func-returns-value]
            _render(data, console)
            return data

        return _runner()

    # ---------------------------------------------------------------------
    # Synchronous path – we already have the data.
    # ---------------------------------------------------------------------
    _render(result, console)  # type: ignore[arg-type]
    return result  # type: ignore[return-value]
