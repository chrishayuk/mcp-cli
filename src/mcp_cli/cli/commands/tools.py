# src/mcp_cli/cli/commands/tools.py
"""
CLI binding for the “tools list” command.

* `tools run`  (Typer entry-point)  - sync, suitable for plain CLI.
* `ToolsListCommand`               - async, used by interactive shell
                                      or CommandRegistry invocations.
"""

from __future__ import annotations

from typing import Any

import typer

# Shared helpers
from mcp_cli.commands.tools import tools_action, tools_action_async
from mcp_cli.tools.manager import get_tool_manager
from mcp_cli.cli.commands.base import BaseCommand

# ─── Typer sub-app ───────────────────────────────────────────────────────────
app = typer.Typer(help="List available tools")


@app.command("run")
def tools_run(
    all: bool = typer.Option(False, "--all", help="Show detailed tool information"),
    raw: bool = typer.Option(False, "--raw", help="Show raw JSON definitions"),
) -> None:
    """
    List unique tools across all connected servers (blocking CLI mode).
    """
    tm = get_tool_manager()
    if tm is None:
        typer.echo("[red]Error:[/] no ToolManager initialized", err=True)
        raise typer.Exit(code=1)

    tools_action(tm, show_details=all, show_raw=raw)
    raise typer.Exit(code=0)


# ─── In-process command for CommandRegistry ─────────────────────────────────
class ToolsListCommand(BaseCommand):
    """`tools list` command (async) for non-interactive invocation."""

    def __init__(self) -> None:
        super().__init__(
            name="tools list",
            help_text="List unique tools across all connected servers.",
        )

    async def execute(self, tool_manager: Any, **params: Any) -> None:
        """
        Delegates to the shared async helper.

        Parameters accepted via **params:
          • all  → bool (show_details)
          • raw  → bool (show_raw)
        """
        show_details = params.get("all", False)
        show_raw = params.get("raw", False)
        await tools_action_async(
            tool_manager,
            show_details=show_details,
            show_raw=show_raw,
        )
