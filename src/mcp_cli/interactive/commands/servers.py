# mcp_cli/interactive/commands/servers.py
from typing import Any, List
from .base import InteractiveCommand
from mcp_cli.commands.servers import servers_action
from mcp_cli.tools.manager import ToolManager

class ServersCommand(InteractiveCommand):
    """Interactive 'servers' command to list connected MCP servers."""

    def __init__(self):
        super().__init__(
            name="servers",
            help_text="List connected servers with their status and tool count.",
            aliases=["srv"],
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager = None,
        **kwargs: Any
    ) -> None:
        """Delegate to the shared servers_action."""
        servers_action(tool_manager)
