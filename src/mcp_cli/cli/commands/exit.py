# mcp_cli/cli/commands/exit.py
import typer
from typing import Any
import logging

# mcp cli imports
from mcp_cli.commands.exit import exit_action
from mcp_cli.cli.commands.base import BaseCommand

# logger
logger = logging.getLogger(__name__)

# ─── Typer sub‐app ───────────────────────────────────────────────────────────
app = typer.Typer(help="Exit the interactive mode")


@app.command("run")
def exit_run() -> None:
    """
    Exit the interactive mode.
    """
    # Perform the shared exit behavior
    exit_action()
    # Then terminate Typer
    raise typer.Exit(code=0)


# ─── In‐process command for CommandRegistry ─────────────────────────────────
class ExitCommand(BaseCommand):
    """`exit` command for non-interactive invocation."""

    def __init__(self):
        super().__init__(
            name="exit", help_text="Exit the interactive mode.", aliases=["quit", "q"]
        )

    async def execute(self, tool_manager: Any, **params: Any) -> bool:
        """
        Delegates to the shared `exit_action`, then returns True to signal exit.
        """
        logger.debug("Executing ExitCommand")
        exit_action()
        return True
