# mcp_cli/commands/resources.py
"""
Shared resources‐listing logic for both interactive and CLI interfaces.
"""
from typing import Any, Dict, List
from rich.console import Console
from rich.table import Table
from mcp_cli.tools.manager import ToolManager

def _human_size(size: int | None) -> str:
    if not size or size < 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

async def resources_action(tm: ToolManager) -> List[Dict[str, Any]]:
    """
    (async) Retrieve resources from the tool manager and render them.
    Returns the raw list of resource dicts.
    """
    console = Console()
    try:
        maybe = tm.list_resources()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return []

    # If it’s a coroutine, await it
    if hasattr(maybe, "__await__"):
        resources = await maybe  # type: ignore
    else:
        resources = maybe  # type: ignore

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
