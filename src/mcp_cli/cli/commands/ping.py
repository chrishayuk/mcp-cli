# mcp_cli/cli/commands/ping.py
import asyncio
import typer
from typing import Dict, List, Optional, Any
import logging

from mcp_cli.commands.ping import ping_action
from mcp_cli.tools.manager import get_tool_manager
from mcp_cli.cli.commands.base import BaseCommand

logger = logging.getLogger(__name__)

# ─── Typer sub‐app ───────────────────────────────────────────────────────────
app = typer.Typer(help="Ping MCP servers")

@app.command("run")
def ping_run(
    server_names: Optional[Dict[int, str]] = typer.Option(
        None, help="Override server display names"
    ),
    targets: List[str] = typer.Argument(None, help="Filter by name/index")
) -> None:
    tm = get_tool_manager()
    if tm is None:
        typer.echo("[red]Error:[/] no ToolManager initialized", err=True)
        raise typer.Exit(code=1)

    success = asyncio.run(ping_action(tm, server_names, targets))
    raise typer.Exit(code=0 if success else 1)


# ─── In‐process CommandRegistry type ─────────────────────────────────────────
class PingCommand(BaseCommand):
    """CLI/interactive ‘ping’ command."""

    def __init__(self):
        super().__init__("ping", "Ping connected MCP servers.")

    async def execute(self, tool_manager: Any, **params: Any) -> bool:
        server_names = params.get("server_names")
        targets = params.get("targets") or []
        logger.debug(f"PingCommand: server_names={server_names}, targets={targets}")
        return await ping_action(tool_manager, server_names, targets)
