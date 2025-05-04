# src/mcp_cli/interactive/commands/tools.py
from typing import Any, List
from .base import InteractiveCommand

# mcp cli
from mcp_cli.commands.tools import tools_action
from mcp_cli.commands.tools_call import tools_call_action
from mcp_cli.tools.manager import ToolManager

class ToolsCommand(InteractiveCommand):
    """Interactive ‘tools’ command: list tools or invoke one."""

    def __init__(self):
        super().__init__(
            name="tools",
            help_text=(
                "List available tools or call one interactively.\n\n"
                "Usage:\n"
                "  tools            List tools\n"
                "  tools --all      Show detailed tool info\n"
                "  tools --raw      Show raw JSON definitions\n"
                "  tools call       Prompt to call a tool"
            ),
            aliases=["t"],
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager = None,
        **kwargs: Any
    ) -> None:
        if args and args[0].lower() == "call":
            # Delegate to the interactive tool-calling flow
            await tools_call_action(tool_manager)
        else:
            # Listing mode
            show_details = "--all" in args
            show_raw     = "--raw" in args
            tools_action(tool_manager, show_details=show_details, show_raw=show_raw)
