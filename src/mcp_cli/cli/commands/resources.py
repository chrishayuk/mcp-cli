# mcp_cli/cli/commands/resources.py
import asyncio
import typer
from typing import Any
import logging

# mcp cli imports
from mcp_cli.commands.resources import resources_action
from mcp_cli.tools.manager import get_tool_manager
from mcp_cli.cli.commands.base import BaseCommand

# logger
logger = logging.getLogger(__name__)

# ─── Typer sub‐app ───────────────────────────────────────────────────────────
app = typer.Typer(help="List connected‐server resources")

@app.command("run")
def resources_run() -> None:
    """
    Show all recorded resources (URI, size, MIME-type) on each server.
    """
    tm = get_tool_manager()
    if tm is None:
        typer.echo("[red]Error:[/] no ToolManager initialized", err=True)
        raise typer.Exit(code=1)

    # Run the async action
    asyncio.run(resources_action(tm))

    raise typer.Exit(code=0)


# ─── In‐process command for CommandRegistry ─────────────────────────────────
class ResourcesListCommand(BaseCommand):
    """`resources list` command for non‐interactive invocation."""

    def __init__(self):
        super().__init__(
            name="resources list",
            help_text="List all resources recorded on connected servers."
        )

    async def execute(self, tool_manager: Any, **params: Any) -> None:
        """
        Delegates to the shared `resources_action`. No additional flags.
        """
        logger.debug("Executing ResourcesListCommand")
        await resources_action(tool_manager)
