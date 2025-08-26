# mcp_cli/cli/commands/ping.py
"""
Ping MCP servers (CLI + registry command).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import typer

from mcp_cli.commands.ping import (
    ping_action,  # sync wrapper (run_blocking)
    ping_action_async,  # real async implementation
)
from mcp_cli.tools.manager import get_tool_manager
from mcp_cli.cli.commands.base import BaseCommand

logger = logging.getLogger(__name__)

# ── Typer sub-app (plain CLI invocation) ────────────────────────────────────
app = typer.Typer(help="Ping connected MCP servers")


@app.command("run")
def ping_run(
    name: List[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Override server display names (index=name)",
    ),
    targets: List[str] = typer.Argument(
        [], metavar="[TARGET]...", help="Filter by server index or name"
    ),
) -> None:
    """
    Blocking CLI entry-point. Examples:

        mcp-cli ping run               # ping all servers
        mcp-cli ping run 0 2           # ping servers 0 and 2
        mcp-cli ping run -n 0=db db    # rename server 0→db and ping “db”
    """
    tm = get_tool_manager()
    if tm is None:
        typer.echo("Error: no ToolManager initialised", err=True)
        raise typer.Exit(code=1)

    mapping: Optional[Dict[int, str]] = None
    if name:
        mapping = {}
        for token in name:
            if "=" in token:
                idx, lbl = token.split("=", 1)
                try:
                    mapping[int(idx)] = lbl
                except ValueError:
                    pass

    ok = ping_action(tm, server_names=mapping, targets=targets)
    raise typer.Exit(code=0 if ok else 1)


# ── Registry / interactive version ──────────────────────────────────────────
class PingCommand(BaseCommand):
    """Global `ping` command (usable from chat / interactive shell)."""

    def __init__(self) -> None:
        super().__init__("ping", "Ping connected MCP servers.")

    async def execute(self, tool_manager: Any, **params: Any) -> bool:  # noqa: D401
        mapping = params.get("server_names")
        targets = params.get("targets", []) or []
        logger.debug("PingCommand: mapping=%s targets=%s", mapping, targets)
        return await ping_action_async(
            tool_manager, server_names=mapping, targets=targets
        )
