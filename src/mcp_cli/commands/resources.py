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

# mcp cli
from mcp_cli.utils.async_utils import run_blocking
from chuk_term.ui import output, format_table
from mcp_cli.context import get_context


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
async def resources_action_async() -> List[Dict[str, Any]]:
    """
    Fetch resources and render a Rich table.

    Returns the raw list to allow callers to re-use the data programmatically.
    """
    # Get context and tool manager
    context = get_context()
    tm = context.tool_manager

    if not tm:
        output.error("No tool manager available")
        return []

    # Most MCP servers expose list_resources() as an awaitable, but some
    # adapters might return a plain list - handle both.
    try:
        maybe = tm.list_resources()
        resources = await maybe if inspect.isawaitable(maybe) else maybe  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        output.error(f"{exc}")
        return []

    resources = resources or []
    if not resources:
        output.info("No resources recorded.")
        return resources

    # Build table data for chuk_term
    table_data = []
    columns = ["Server", "URI", "Size", "MIME-type"]

    for item in resources:
        table_data.append(
            {
                "Server": item.get("server", "-"),
                "URI": item.get("uri", "-"),
                "Size": _human_size(item.get("size")),
                "MIME-type": item.get("mimeType", "-"),
            }
        )

    # Create and display table using chuk-term
    table = format_table(table_data, title="Resources", columns=columns)
    output.print_table(table)
    return resources


# ════════════════════════════════════════════════════════════════════════
# sync wrapper - used by non-interactive CLI paths
# ════════════════════════════════════════════════════════════════════════
def resources_action() -> List[Dict[str, Any]]:
    """
    Blocking wrapper around :pyfunc:`resources_action_async`.

    Raises *RuntimeError* if called from inside an active event-loop.
    """
    return run_blocking(resources_action_async())


__all__ = [
    "resources_action_async",
    "resources_action",
]
