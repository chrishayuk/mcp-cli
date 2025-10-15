# src/mcp_cli/commands/actions/prompts.py
"""
Prompts action for MCP CLI.

List prompt templates from connected MCP servers.

Public functions:
* **prompts_action_async()** - Async function to list prompts.
* **prompts_action()** - Sync wrapper for legacy CLI paths.
* **prompts_action_cmd()** - Alias for backward compatibility.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

from mcp_cli.utils.async_utils import run_blocking
from chuk_term.ui import output, format_table
from mcp_cli.context import get_context


async def prompts_action_async() -> List[Dict[str, Any]]:
    """
    Fetch and display prompt templates from all connected MCP servers.

    Returns:
        List of prompt dictionaries from all servers.
    """
    context = get_context()
    tm = context.tool_manager

    if not tm:
        output.error("No tool manager available")
        return []

    try:
        maybe = tm.list_prompts()
        prompts = await maybe if inspect.isawaitable(maybe) else maybe
    except Exception as exc:  # noqa: BLE001
        output.error(f"{exc}")
        return []

    prompts = prompts or []
    if not prompts:
        output.info("No prompts recorded.")
        return []

    # Build table data
    table_data = []
    columns = ["Server", "Name", "Description"]

    for item in prompts:
        table_data.append(
            {
                "Server": item.get("server", "-"),
                "Name": item.get("name", "-"),
                "Description": item.get("description", "-"),
            }
        )

    # Display table
    table = format_table(table_data, title="Prompts", columns=columns)
    output.print_table(table)
    return list(prompts)


def prompts_action() -> List[Dict[str, Any]]:
    """
    Sync wrapper for prompts_action_async.

    Returns:
        List of prompt dictionaries from all servers.

    Raises:
        RuntimeError: If called from inside an active event loop.
    """
    return run_blocking(prompts_action_async())


async def prompts_action_cmd() -> List[Dict[str, Any]]:
    """
    Alias for prompts_action_async (backward compatibility).

    Returns:
        List of prompt dictionaries from all servers.
    """
    return await prompts_action_async()


__all__ = [
    "prompts_action_async",
    "prompts_action",
    "prompts_action_cmd",
]
