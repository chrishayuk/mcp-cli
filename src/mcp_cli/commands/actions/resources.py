# src/mcp_cli/commands/actions/resources.py
"""
Resources action for MCP CLI.

List resources (files, blobs, artifacts) from connected MCP servers.

Public functions:
* **resources_action_async()** - Async function to list resources.
* **resources_action()** - Sync wrapper for legacy CLI paths.
"""

from __future__ import annotations

import inspect
from typing import List

from mcp_cli.utils.async_utils import run_blocking
from chuk_term.ui import output, format_table
from mcp_cli.context import get_context
from mcp_cli.commands.models import ResourceInfoResponse


def _human_size(size: int | None) -> str:
    """Convert size in bytes to human-readable string (KB/MB/GB)."""
    if size is None or size < 0:
        return "-"
    current_size: float = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if current_size < 1024:
            return f"{current_size:.0f} {unit}"
        current_size = current_size / 1024
    return f"{current_size:.1f} TB"


async def resources_action_async() -> List[ResourceInfoResponse]:
    """
    Fetch and display resources from all connected MCP servers.

    Returns:
        List of resource response models from all servers.
    """
    context = get_context()
    tm = context.tool_manager

    if not tm:
        output.error("No tool manager available")
        return []

    try:
        maybe = tm.list_resources()
        resources = await maybe if inspect.isawaitable(maybe) else maybe
    except Exception as exc:  # noqa: BLE001
        output.error(f"{exc}")
        return []

    resources = resources or []
    if not resources:
        output.info("No resources recorded.")
        return []

    # Convert to Pydantic models
    resource_models = []
    table_data = []
    columns = ["Server", "URI", "Size", "MIME-type"]

    for item in resources:
        # Create Pydantic model
        resource_model = ResourceInfoResponse(
            uri=item.get("uri", "-"),
            name=item.get("name"),
            description=item.get("description"),
            mime_type=item.get("mimeType"),
            server=item.get("server", "-"),
        )
        resource_models.append(resource_model)

        # Build table row
        table_data.append(
            {
                "Server": resource_model.server,
                "URI": resource_model.uri,
                "Size": _human_size(item.get("size")),
                "MIME-type": resource_model.mime_type or "-",
            }
        )

    # Display table
    table = format_table(table_data, title="Resources", columns=columns)
    output.print_table(table)
    return resource_models


def resources_action() -> List[ResourceInfoResponse]:
    """
    Sync wrapper for resources_action_async.

    Returns:
        List of resource dictionaries from all servers.

    Raises:
        RuntimeError: If called from inside an active event loop.
    """
    return run_blocking(resources_action_async())


__all__ = [
    "resources_action_async",
    "resources_action",
]
