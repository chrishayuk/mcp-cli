# mcp_cli/interactive/commands/resources.py
from typing import Any, List
from .base import InteractiveCommand
from mcp_cli.commands.resources import resources_action
from mcp_cli.tools.manager import ToolManager

class ResourcesCommand(InteractiveCommand):
    """Interactive 'resources' command to list available resources."""

    def __init__(self):
        super().__init__(
            name="resources",
            help_text="List available resources from all connected servers.",
            aliases=["res"],
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager = None,
        **kwargs: Any
    ) -> Any:
        # delegates to the shared async action
        return await resources_action(tool_manager)
