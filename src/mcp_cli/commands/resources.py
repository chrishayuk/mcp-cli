# src/mcp_cli/commands/resources.py
"""
Shared resources-listing logic for both interactive and CLI interfaces.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking


def _human_size(size: int | None) -> str:
    if not size or size < 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


async def resources_action_async(tm: ToolManager) -> List[Dict[str, Any]]:
    console = Console()
    try:
        maybe = tm.list_resources()
        try:
            resources = await maybe if inspect.isawaitable(maybe) else maybe
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            return []
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return []
    resources = resources or []
    if not resources:
        console.print("[dim]No resources recorded.[/dim]")
        return resources
    table = Table(title="Resources", header_style="bold magenta")
    table.add_column("Server", style="cyan")
    table.add_column("URI", style="yellow")
    table.add_column("Size", justify="right")
    table.add_column("MIME-type")
    for item in resources:
        table.add_row(
            item.get("server", "-"),
            item.get("uri", "-"),
            _human_size(item.get("size")),
            item.get("mimeType", "-"),
        )
    console.print(table)
    return resources


def resources_action(tm: ToolManager) -> List[Dict[str, Any]]:
    return run_blocking(resources_action_async(tm))
