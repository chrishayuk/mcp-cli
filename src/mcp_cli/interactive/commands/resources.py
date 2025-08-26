# mcp_cli/interactive/commands/resources.py
"""
Interactive **resources** command - list every resource discovered by the
connected MCP servers (URI, size, MIME-type, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, List

from chuk_term.ui import output  # ← NEW
from mcp_cli.commands.resources import resources_action_async  # shared async helper
from mcp_cli.tools.manager import ToolManager
from .base import InteractiveCommand

log = logging.getLogger(__name__)


class ResourcesCommand(InteractiveCommand):
    """Display resources harvested by all connected servers."""

    def __init__(self) -> None:
        super().__init__(
            name="resources",
            aliases=["res"],
            help_text="List resources (URI, size, MIME-type) on connected servers.",
        )

    # ------------------------------------------------------------------
    async def execute(  # noqa: D401  (simple entry-point)
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **_: Any,
    ) -> None:
        if tool_manager is None:
            output.print("[red]Error:[/red] ToolManager not available.")
            log.debug("ResourcesCommand triggered without a ToolManager instance.")
            return

        # currently no extra flags - reserved for future
        await resources_action_async(tool_manager)
