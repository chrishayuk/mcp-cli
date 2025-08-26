# mcp_cli/interactive/commands/tools.py
"""
Interactive **tools** command - list all tools or launch the “call-a-tool”
helper inside the *interactive shell* (not the chat TUI).
"""

from __future__ import annotations

import logging
from typing import Any, List

from chuk_term.ui import output  # ← NEW
from mcp_cli.commands.tools import tools_action_async  # shared async helper
from mcp_cli.commands.tools_call import tools_call_action
from mcp_cli.tools.manager import ToolManager
from .base import InteractiveCommand

log = logging.getLogger(__name__)


class ToolsCommand(InteractiveCommand):
    """Show available tools or invoke one interactively."""

    def __init__(self) -> None:
        super().__init__(
            name="tools",
            aliases=["t"],
            help_text=(
                "List available tools or run one interactively.\n\n"
                "Usage:\n"
                "  tools              - list tools\n"
                "  tools --all        - include parameter details\n"
                "  tools --raw        - dump raw JSON\n"
                "  tools call         - open interactive call helper"
            ),
        )

    # ────────────────────────────────────────────────────────────────
    async def execute(  # noqa: D401
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **_: Any,
    ) -> None:
        # Ensure ToolManager exists
        if tool_manager is None:
            output.print("[red]Error:[/red] ToolManager not available.")
            log.debug("ToolsCommand executed without a ToolManager instance.")
            return

        # "tools call" → interactive call helper
        if args and args[0].lower() == "call":
            await tools_call_action(tool_manager)
            return

        # Otherwise list tools
        show_details = "--all" in args
        show_raw = "--raw" in args
        await tools_action_async(
            tool_manager,
            show_details=show_details,
            show_raw=show_raw,
        )
