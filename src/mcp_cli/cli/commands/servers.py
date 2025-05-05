# src/mcp_cli/cli/commands/servers.py
import typer
from typing import Any, Optional, Dict, List
import logging

# mcp cli imports
from mcp_cli.commands.servers import servers_action
from mcp_cli.tools.manager import get_tool_manager
from mcp_cli.cli.commands.base import BaseCommand

# logger
logger = logging.getLogger(__name__)

# ─── Typer sub‐app ───────────────────────────────────────────────────────────
app = typer.Typer(help="List connected MCP servers")

@app.command("run")
def servers_run() -> None:
    """
    Show all connected servers with their status and tool counts.
    """
    tm = get_tool_manager()
    if tm is None:
        typer.echo("[red]Error:[/] no ToolManager initialized", err=True)
        raise typer.Exit(code=1)

    # Call the shared action
    servers_action(tm)
    raise typer.Exit(code=0)


# ─── In‐process command for CommandRegistry ─────────────────────────────────
class ServersListCommand(BaseCommand):
    """`servers list` command for non‐interactive invocation."""

    def __init__(self):
        super().__init__(
            name="servers list",
            help_text="Show all connected servers with their status and tool counts."
        )

    async def execute(self, tool_manager: Any, **params: Any) -> None:
        """
        Delegates to the shared `servers_action`. No additional flags.
        """
        logger.debug("Executing ServersListCommand")
        servers_action(tool_manager)
