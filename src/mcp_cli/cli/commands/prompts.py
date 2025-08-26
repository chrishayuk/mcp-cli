# src/mcp_cli/cli/prompts.py
"""
CLI binding for “prompts list”.
"""

from __future__ import annotations

import logging
from typing import Any, List

import typer

from mcp_cli.commands.prompts import prompts_action, prompts_action_async
from mcp_cli.tools.manager import get_tool_manager
from mcp_cli.cli.commands.base import BaseCommand

logger = logging.getLogger(__name__)

# ─── Typer sub-app (plain CLI) ───────────────────────────────────────────────
app = typer.Typer(help="List prompts recorded on connected MCP servers")


@app.command("run")
def prompts_run() -> None:
    """
    List all prompts (blocking CLI mode).
    """
    tm = get_tool_manager()
    if tm is None:
        typer.echo("Error: no ToolManager initialised", err=True)
        raise typer.Exit(code=1)

    prompts_action(tm)
    raise typer.Exit(code=0)


# ─── Registry / interactive variant ─────────────────────────────────────────
class PromptsListCommand(BaseCommand):
    """`prompts list` command usable from interactive shell or scripts."""

    def __init__(self) -> None:
        super().__init__(
            name="prompts list",
            help_text="List all prompts recorded on connected servers.",
        )

    async def execute(self, tool_manager: Any, **_: Any) -> None:
        logger.debug("Executing PromptsListCommand")
        await prompts_action_async(tool_manager)

    # ------------------------------------------------------------------
    # Typer registration for the CommandRegistry wrapper
    # ------------------------------------------------------------------
    def register(self, app: typer.Typer, run_command_func) -> None:
        """
        Add a `prompts list` sub-command to the given Typer app.

        A variadic *kwargs* argument is optional so users are not forced
        to provide key=value pairs.
        """

        @app.command("list")
        def _prompts_list(
            # global CLI options
            config_file: str = "server_config.json",
            server: str | None = None,
            provider: str = "openai",
            model: str | None = None,
            disable_filesystem: bool = False,
            # optional key=value extras
            kwargs: List[str] = typer.Argument(
                [], metavar="KWARGS", help="Extra key=value arguments"
            ),
        ) -> None:  # noqa: D401 (Typer callback)
            from mcp_cli.cli_options import process_options

            servers, _, server_names = process_options(
                server, disable_filesystem, provider, model, config_file
            )

            # Forward to the shared async execute method through the
            # registry’s run_command helper.
            run_command_func(
                self.wrapped_execute,
                config_file,
                servers,
                extra_params={"server_names": server_names},
            )
