import asyncio
import typer
from typing import Any
import logging

# Shared prompts implementation
from mcp_cli.commands.prompts import prompts_action
from mcp_cli.tools.manager import get_tool_manager

# BaseCommand for in-process registry
from mcp_cli.cli.commands.base import BaseCommand

logger = logging.getLogger(__name__)

# ─── Typer sub-app ───────────────────────────────────────────────────────────
app = typer.Typer(help="List recorded prompts")

@app.command("run")
def prompts_run() -> None:
    """
    List all prompts recorded on connected servers.
    """
    tm = get_tool_manager()
    if tm is None:
        typer.echo("[red]Error:[/] no ToolManager initialized", err=True)
        raise typer.Exit(code=1)

    # Run the async action in this sync context
    asyncio.run(prompts_action(tm))
    raise typer.Exit(code=0)


# ─── In-process command for CommandRegistry ─────────────────────────────────
class PromptsListCommand(BaseCommand):
    """`prompts list` command for non-interactive invocation."""

    def __init__(self):
        super().__init__(
            name="prompts list",
            help_text="List all prompts recorded on connected servers."
        )

    async def execute(self, tool_manager: Any, **params: Any) -> None:
        """
        Delegates to the shared `prompts_action`. No additional flags are required.
        """
        logger.debug("Executing PromptsListCommand")
        # properly await the shared async action
        await prompts_action(tool_manager)
