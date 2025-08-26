# src/mcp_cli/commands/resources.py
"""
List binary *resources* (files, blobs, artefacts) known to every connected
MCP server.

There are three public call-sites:

* **resources_action_async(tm)** - canonical coroutine for chat / TUI.
* **resources_action(tm)**       - tiny sync wrapper for legacy CLI paths.
* **_human_size(n)**             - helper to pretty-print bytes.

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
# helpers
# ════════════════════════════════════════════════════════════════════════
def _human_size(size: int | None) -> str:
    """Convert *size* in bytes to a human-readable string (KB/MB/GB)."""
    if size is None or size < 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ════════════════════════════════════════════════════════════════════════
# async (primary) implementation
# ════════════════════════════════════════════════════════════════════════
async def resources_action_async(tm: ToolManager) -> List[Dict[str, Any]]:
    """
    Fetch resources from *tm* and render a Rich table.

    Returns the raw list to allow callers to re-use the data programmatically.
    """

    # Most MCP servers expose list_resources() as an awaitable, but some
    # adapters might return a plain list - handle both.
    try:
        maybe = tm.list_resources()
        resources = await maybe if inspect.isawaitable(maybe) else maybe  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        output.print(f"[red]Error:[/red] {exc}")
        return []

    resources = resources or []
    if not resources:
        output.print("[dim]No resources recorded.[/dim]")
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

    output.print(table)
    return resources


# ════════════════════════════════════════════════════════════════════════
# sync wrapper - used by non-interactive CLI paths
# ════════════════════════════════════════════════════════════════════════
def resources_action(tm: ToolManager) -> List[Dict[str, Any]]:
    """
    Blocking wrapper around :pyfunc:`resources_action_async`.

    Raises *RuntimeError* if called from inside an active event-loop.
    """
    return run_blocking(resources_action_async(tm))


__all__ = [
    "resources_action_async",
    "resources_action",
]
