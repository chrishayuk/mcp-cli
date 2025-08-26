# mcp_cli/cli/commands/tools_call.py
from __future__ import annotations
import asyncio
import typer
from typing import Any
import logging

# shared action
from mcp_cli.commands.tools_call import tools_call_action
from mcp_cli.tools.manager import get_tool_manager

# BaseCommand for registry
from mcp_cli.cli.commands.base import BaseCommand

logger = logging.getLogger(__name__)

# ─── Typer sub‐app ───────────────────────────────────────────────────────────
app = typer.Typer(help="Call a specific tool with arguments")


@app.command("run")
def tools_call_run() -> None:
    """
    Launch the interactive tool‐call interface.
    """
    tm = get_tool_manager()
    if tm is None:
        typer.echo("[red]Error:[/] no ToolManager initialized", err=True)
        raise typer.Exit(code=1)

    # Run the async action
    asyncio.run(tools_call_action(tm))

    # Exit successfully
    raise typer.Exit(code=0)


# ─── In‐process command for CommandRegistry ─────────────────────────────────
class ToolsCallCommand(BaseCommand):
    """`tools call` command for non‐interactive invocation."""

    def __init__(self):
        super().__init__(
            name="tools call", help_text="Call a specific tool with arguments."
        )

    async def execute(self, tool_manager: Any, **params: Any) -> None:
        """
        Delegates to the shared `tools_call_action`.
        """
        logger.debug("Executing ToolsCallCommand")
        await tools_call_action(tool_manager)
