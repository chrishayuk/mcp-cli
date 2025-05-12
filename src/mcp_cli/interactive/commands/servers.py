# mcp_cli/interactive/commands/servers.py
"""
Interactive “servers” command - list connected MCP servers.
"""
from __future__ import annotations

from typing import Any, List

from rich import print

from .base import InteractiveCommand
from mcp_cli.commands.servers import servers_action_async
from mcp_cli.tools.manager import ToolManager


class ServersCommand(InteractiveCommand):
    """List connected MCP servers and their status/tool count."""

    def __init__(self) -> None:
        super().__init__(
            name="servers",
            help_text="List connected servers with their status and tool count.",
            aliases=["srv"],
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **_: Any,
    ) -> None:
        if not tool_manager:
            print("[red]Error:[/red] ToolManager not available.")
            return

        await servers_action_async(tool_manager)
