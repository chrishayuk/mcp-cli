# src/mcp_cli/commands/prompts.py
"""
List stored *prompts* on every connected MCP server.

There are three public call-sites:

* **prompts_action_async(tm)**   – canonical async implementation.
* **prompts_action(tm)**         – sync wrapper for plain CLI commands.
* **prompts_action_cmd(tm)**     – thin alias for interactive `/prompts`
  so the chat UI can `await` directly without hitting run_blocking().
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking


# ──────────────────────────────────────────────────────────────────
# async (primary) implementation
# ──────────────────────────────────────────────────────────────────
async def prompts_action_async(tm: ToolManager) -> List[Dict[str, Any]]:
    console = Console()

    try:
        maybe = tm.list_prompts()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        return []

    prompts = await maybe if inspect.isawaitable(maybe) else maybe
    prompts = prompts or []
    if not prompts:
        console.print("[dim]No prompts recorded.[/dim]")
        return prompts

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
    return prompts


# ──────────────────────────────────────────────────────────────────
# sync wrapper – used by *non-interactive* CLI commands
# ──────────────────────────────────────────────────────────────────
def prompts_action(tm: ToolManager) -> List[Dict[str, Any]]:
    """
    Blocking helper so legacy CLI commands can remain synchronous.
    Raises a RuntimeError if invoked from inside a running event-loop.
    """
    return run_blocking(prompts_action_async(tm))


# ──────────────────────────────────────────────────────────────────
# alias for chat/interactive mode
# ──────────────────────────────────────────────────────────────────
async def prompts_action_cmd(tm: ToolManager) -> List[Dict[str, Any]]:
    """
    Alias exported for the interactive `/prompts` command.

    The chat UI runs inside an event-loop already, so it should import and
    `await` this coroutine directly instead of using the sync wrapper above.
    """
    return await prompts_action_async(tm)


__all__ = [
    "prompts_action_async",
    "prompts_action",
    "prompts_action_cmd",
]
