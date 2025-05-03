# mcp_cli/commands/resources.py
"""
Show every resource exposed by the connected MCP servers.

The function mirrors *prompts_list()* – see that module’s doc-string for
details.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from mcp_cli.tools.manager import ToolManager, get_tool_manager


def _human_size(size: int | None) -> str:
    if not size or size < 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _render(res: List[Dict[str, Any]], console: Console) -> None:
    if not res:
        console.print("[dim]No resources recorded.[/dim]")
        return

    table = Table(title="Resources", header_style="bold magenta")
    table.add_column("Server", style="cyan")
    table.add_column("URI", style="yellow")
    table.add_column("Size", justify="right")
    table.add_column("MIME-type")

    for item in res:
        table.add_row(
            item.get("server", "-"),
            item.get("uri", "-"),
            _human_size(item.get("size")),
            item.get("mimeType", "-"),
        )

    console.print(table)


def resources_list(
    *,
    stream_manager: Optional[ToolManager] = None,
    console: Optional[Console] = None,
) -> List[Dict[str, Any]] | asyncio.Future:
    """
    List every resource from every server.

    Works in both synchronous and asynchronous contexts – see
    *prompts_list()* for the reasoning.
    """
    console = console or Console()
    tm = stream_manager or get_tool_manager()
    if tm is None:
        console.print("[red]Error:[/red] no ToolManager available")
        return []

    result = tm.list_resources()
    if inspect.isawaitable(result):
        async def _runner() -> List[Dict[str, Any]]:
            data = await result  # type: ignore[func-returns-value]
            _render(data, console)
            return data

        return _runner()

    _render(result, console)  # type: ignore[arg-type]
    return result  # type: ignore[return-value]
