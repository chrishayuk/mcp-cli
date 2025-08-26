# mcp_cli/interactive/commands/ping.py
"""
Interactive **ping** command - measure round-trip latency to each
connected MCP server.

Usage
-----
  ping               → ping every server
  ping 0 api         → ping only server #0 and the one named “api”
  pg …               → short alias
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from chuk_term.ui import output  # ← NEW
from mcp_cli.commands.ping import ping_action_async  # shared async helper
from mcp_cli.tools.manager import ToolManager
from .base import InteractiveCommand

log = logging.getLogger(__name__)


class PingCommand(InteractiveCommand):
    """Measure server latency (interactive shell)."""

    def __init__(self) -> None:
        super().__init__(
            name="ping",
            aliases=["pg"],  # handy two-letter shortcut
            help_text="Ping each MCP server (optionally filter by index or name).",
        )

    # ------------------------------------------------------------------
    async def execute(  # noqa: D401
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **ctx: Dict[str, Any],
    ) -> None:
        """
        Delegate to :func:`mcp_cli.commands.ping.ping_action_async`.

        *args* contains any filters supplied after the command word.
        """

        if tool_manager is None:
            log.debug("PingCommand executed without ToolManager - aborting.")
            output.print("[red]Error:[/red] ToolManager not available.")
            return

        server_names = ctx.get("server_names")  # may be None
        targets = args  # filters (index / partial name)

        await ping_action_async(
            tool_manager,
            server_names=server_names,
            targets=targets,
        )
