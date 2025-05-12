# src/mcp_cli/commands/servers.py
"""
Shared servers-listing logic for both interactive and CLI interfaces.
"""
from __future__ import annotations

import asyncio
from rich.console import Console
from rich.table import Table

from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking


# ──────────────────────────────────────────────────────────────────
# async (canonical) version
# ──────────────────────────────────────────────────────────────────
async def servers_action_async(tm: ToolManager) -> None:
    """
    Fetch connected servers via ToolManager and render a table.
    """
    server_info = await tm.get_server_info()
    console = Console()

    if not server_info:
        console.print("[yellow]No servers connected.[/yellow]")
        return

    table = Table(title="Connected Servers")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Tools", style="cyan", justify="right")
    table.add_column("Status", style="green")

    for srv in server_info:
        table.add_row(
            str(srv.id),
            srv.name,
            str(srv.tool_count),
            srv.status,
        )

    console.print(table)


# ──────────────────────────────────────────────────────────────────
# sync helper – legacy entry-points can keep using this
# ──────────────────────────────────────────────────────────────────
def servers_action(tm: ToolManager) -> None:
    run_blocking(servers_action_async(tm))
