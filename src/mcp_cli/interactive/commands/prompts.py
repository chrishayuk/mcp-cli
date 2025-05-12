# mcp_cli/interactive/commands/prompts.py
"""
Interactive “prompts” command - lists stored prompts on all connected servers.
"""
from __future__ import annotations

from typing import Any, List

from mcp_cli.commands.prompts import prompts_action_cmd  # ← async helper
from mcp_cli.tools.manager import ToolManager
from .base import InteractiveCommand


class PromptsCommand(InteractiveCommand):
    """List available prompts."""

    def __init__(self) -> None:
        super().__init__(
            name="prompts",
            help_text="List available prompts from all connected servers.",
            aliases=["p"],
        )

    # ------------------------------------------------------------------
    # InteractiveCommand interface
    # ------------------------------------------------------------------
    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **_: Any,
    ) -> None:
        if not tool_manager:
            from rich import print
            print("[red]Error:[/red] ToolManager not available.")
            return

        await prompts_action_cmd(tool_manager)
