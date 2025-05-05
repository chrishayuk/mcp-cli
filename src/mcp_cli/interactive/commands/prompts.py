# mcp_cli/interactive/commands/prompts.py
from typing import Any, List
from .base import InteractiveCommand
from mcp_cli.commands.prompts import prompts_action
from mcp_cli.tools.manager import ToolManager

class PromptsCommand(InteractiveCommand):
    """Interactive 'prompts' command to list available prompts."""

    def __init__(self):
        super().__init__(
            name="prompts",
            help_text="List available prompts from all connected servers.",
            aliases=["p"],
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager = None,
        **kwargs: Any
    ) -> Any:
        # Properly await the shared async action
        return await prompts_action(tool_manager)
