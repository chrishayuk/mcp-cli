# mcp_cli/interactive/commands/tools.py
"""
Interactive “tools” command – list tools or open the interactive call helper.
"""
from __future__ import annotations

from typing import Any, List

from rich import print

from .base import InteractiveCommand
from mcp_cli.commands.tools import tools_action_async         # ← async helper
from mcp_cli.commands.tools_call import tools_call_action
from mcp_cli.tools.manager import ToolManager


class ToolsCommand(InteractiveCommand):
    """List available tools, or interactively invoke one."""

    def __init__(self) -> None:
        super().__init__(
            name="tools",
            aliases=["t"],
            help_text=(
                "List available tools or call one interactively.\n\n"
                "Usage:\n"
                "  tools              List tools\n"
                "  tools --all        Show parameter details\n"
                "  tools --raw        Show raw JSON definitions\n"
                "  tools call         Prompt to call a tool"
            ),
        )

    # ------------------------------------------------------------------
    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **_: Any,
    ) -> None:
        if not tool_manager:
            print("[red]Error:[/red] ToolManager not available.")
            return

        # 'tools call' → open interactive tool-caller
        if args and args[0].lower() == "call":
            await tools_call_action(tool_manager)
            return

        # otherwise list tools
        show_details = "--all" in args
        show_raw     = "--raw" in args
        await tools_action_async(
            tool_manager,
            show_details=show_details,
            show_raw=show_raw,
        )
