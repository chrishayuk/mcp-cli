# mcp_cli/interactive/commands/servers.py
"""
Interactive **servers** command - display every connected MCP server with
its status and tool count.
"""
from __future__ import annotations

import logging
from typing import Any, List

from mcp_cli.utils.rich_helpers import get_console           # ← NEW
from mcp_cli.commands.servers import servers_action_async    # shared helper
from mcp_cli.tools.manager import ToolManager
from .base import InteractiveCommand

log = logging.getLogger(__name__)


class ServersCommand(InteractiveCommand):
    """Show connected servers and their basic stats."""

    def __init__(self) -> None:
        super().__init__(
            name="servers",
            aliases=["srv"],
            help_text="List connected MCP servers with status and tool count.",
        )

    # ────────────────────────────────────────────────────────────────
    async def execute(                       # noqa: D401
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **_: Any,
    ) -> None:
        console = get_console()

        if tool_manager is None:
            console.print("[red]Error:[/red] ToolManager not available.")
            log.debug("ServersCommand executed without a ToolManager instance.")
            return

        # No extra arguments are currently supported but kept for future use
        await servers_action_async(tool_manager)
