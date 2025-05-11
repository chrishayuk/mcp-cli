# mcp_cli/interactive/commands/resources.py
"""
Interactive “resources” command – list resources on connected servers.
"""
from __future__ import annotations

from typing import Any, List

from .base import InteractiveCommand
from mcp_cli.commands.resources import resources_action_async  # ← async helper
from mcp_cli.tools.manager import ToolManager
from rich import print


class ResourcesCommand(InteractiveCommand):
    """List available resources from all connected servers."""

    def __init__(self) -> None:
        super().__init__(
            name="resources",
            help_text="List resources recorded by connected servers.",
            aliases=["res"],
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
        await resources_action_async(tool_manager)
