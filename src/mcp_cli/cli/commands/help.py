# src/mcp_cli/cli/commands/help.py

import typer
from typing import Optional, Any
import logging


# shared implementation
from mcp_cli.commands.help import help_action

# BaseCommand for in-process registry
from mcp_cli.cli.commands.base import BaseCommand

logger = logging.getLogger(__name__)

# ─── Typer sub-app ────────────────────────────────────────────────────────────
app = typer.Typer(help="Display interactive-command help")


@app.command("run")
def help_run(
    command: Optional[str] = typer.Argument(
        None, help="Name of the command to show detailed help for"
    ),
) -> None:
    """
    Show help for all commands, or detailed help for one command.
    """
    help_action(command)
    # Typer will auto-exit after returning from this function.


# ─── In-process command for CommandRegistry ─────────────────────────────────
class HelpCommand(BaseCommand):
    """`help` command for non-interactive invocation."""

    def __init__(self):
        super().__init__(
            name="help",
            help_text="Display available commands or help for a specific command.",
        )

    async def execute(self, tool_manager: Any, **params: Any) -> None:
        """
        Delegates to the shared `help_action`.
        Expects optional:
          • command: str
        """
        command = params.get("command")
        logger.debug(f"Executing HelpCommand for: {command!r}")
        help_action(command)
