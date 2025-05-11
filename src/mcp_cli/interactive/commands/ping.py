# mcp_cli/interactive/commands/ping.py
"""
Interactive “ping” command – measure latency to connected MCP servers.
"""
from __future__ import annotations

from typing import Any, List

from rich import print

from .base import InteractiveCommand
from mcp_cli.commands.ping import ping_action_async        # ← async helper
from mcp_cli.tools.manager import ToolManager


class PingCommand(InteractiveCommand):
    """Ping connected servers (optionally filter by index or name)."""

    def __init__(self) -> None:
        super().__init__(
            name="ping",
            help_text="Ping connected servers (optionally filter by index/name).",
            aliases=[],
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **ctx: Any,
    ) -> bool:
        if not tool_manager:
            print("[red]Error:[/red] ToolManager not available.")
            return False

        server_names = ctx.get("server_names")  # may be None
        targets = args  # args exclude the command word itself
        return await ping_action_async(
            tool_manager,
            server_names=server_names,
            targets=targets,
        )
