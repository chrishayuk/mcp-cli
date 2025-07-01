# src/mcp_cli/commands/servers.py
"""
Show a table of *connected MCP servers* and how many tools each exposes.

Three public entry-points
-------------------------
* **servers_action_async(tm)** - primary coroutine for chat / TUI.
* **servers_action(tm)**       - thin sync wrapper for legacy CLI paths.
* The module-level doc-string itself gives a short description that the
  `/help` command will pick up.
"""
from __future__ import annotations
from typing import List
from rich.table import Table

# mcp cli
from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking
from mcp_cli.utils.rich_helpers import get_console


# ════════════════════════════════════════════════════════════════════════
# async (canonical) implementation
# ════════════════════════════════════════════════════════════════════════
async def servers_action_async(tm: ToolManager) -> List:  # noqa: D401
    """
    Retrieve server metadata from *tm* and render a Rich table.

    Returns the raw list so callers may re-use the data programmatically.
    
    Note: Logging noise is controlled by the centralized logging configuration
    in mcp_cli.logging_config.setup_logging().
    """
    console = get_console()
    server_info = await tm.get_server_info()

    if not server_info:
        console.print("[yellow]No servers connected.[/yellow]")
        return server_info

    table = Table(title="Connected Servers", header_style="bold magenta")
    table.add_column("ID",    style="cyan")
    table.add_column("Name",  style="green")
    table.add_column("Tools", style="cyan", justify="right")
    table.add_column("Status")

    for srv in server_info:
        table.add_row(
            str(srv.id),
            srv.name,
            str(srv.tool_count),
            srv.status,
        )

    console.print(table)
    return server_info


# ════════════════════════════════════════════════════════════════════════
# sync helper - kept for non-async CLI paths
# ════════════════════════════════════════════════════════════════════════
def servers_action(tm: ToolManager) -> List:
    """Blocking wrapper around :pyfunc:`servers_action_async`."""
    return run_blocking(servers_action_async(tm))


__all__ = ["servers_action_async", "servers_action"]