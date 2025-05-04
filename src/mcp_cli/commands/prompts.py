# mcp_cli/commands/prompts.py
"""
Shared prompts‐listing logic for both interactive and CLI interfaces.
"""

import inspect
from typing import Any, Dict, List
from rich.console import Console
from rich.table import Table
from mcp_cli.tools.manager import ToolManager


async def prompts_action(tm: ToolManager) -> List[Dict[str, Any]]:
    """
    (async) Retrieve prompts from the tool manager and render them.
    Returns the raw list of prompt dicts.
    """
    console = Console()
    try:
        maybe = tm.list_prompts()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return []

    # If `list_prompts` returned a coroutine, await it
    if inspect.isawaitable(maybe):
        prompts = await maybe  # type: ignore
    else:
        prompts = maybe  # type: ignore

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
