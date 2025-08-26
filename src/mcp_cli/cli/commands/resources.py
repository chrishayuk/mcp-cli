# src/mcp_cli/cli/commands/resources.py
"""
CLI binding for “resources list”.
"""

from __future__ import annotations

import logging
from typing import Any

import typer

# shared helpers
from mcp_cli.commands.resources import resources_action, resources_action_async
from mcp_cli.tools.manager import get_tool_manager
from mcp_cli.cli.commands.base import BaseCommand

logger = logging.getLogger(__name__)

# ─── Typer sub-app ───────────────────────────────────────────────────────────
app = typer.Typer(help="List resources recorded on connected MCP servers")


@app.command("run")
def resources_run() -> None:
    """
    Show all recorded resources (blocking CLI mode).
    """
    tm = get_tool_manager()
    if tm is None:
        typer.echo("Error: no ToolManager initialised", err=True)
        raise typer.Exit(code=1)

    resources_action(tm)  # synchronous wrapper
    raise typer.Exit(code=0)


# ─── In-process command for CommandRegistry ─────────────────────────────────
class ResourcesListCommand(BaseCommand):
    """`resources list` command usable from interactive shell or scripts."""

    def __init__(self) -> None:
        super().__init__(
            name="resources list",
            help_text="List all resources recorded on connected servers.",
        )

    async def execute(self, tool_manager: Any, **_: Any) -> None:
        """
        Delegates to the shared *async* helper.
        """
        logger.debug("Executing ResourcesListCommand")
        await resources_action_async(tool_manager)
