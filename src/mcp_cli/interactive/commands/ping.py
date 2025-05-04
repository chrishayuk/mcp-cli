# mcp_cli/interactive/commands/ping.py
from typing import Any, List
from .base import InteractiveCommand
from mcp_cli.commands.ping import ping_action

class PingCommand(InteractiveCommand):
    """Interactive-mode 'ping' command."""

    def __init__(self):
        super().__init__(
            name="ping",
            help_text="Ping connected servers (optionally filter by index/name).",
            aliases=[]
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: Any = None,
        **kwargs: Any
    ) -> bool:
        server_names = kwargs.get("server_names")
        return await ping_action(tool_manager, server_names, args)
