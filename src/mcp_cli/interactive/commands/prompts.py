# mcp_cli/interactive/commands/prompts.py
"""
Interactive **prompts** command - list prompt templates stored on every
connected MCP server.

Usage inside the shell
----------------------
  prompts          → show a table of prompts
  pr               → short alias
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from chuk_term.ui import output  # ← NEW
from mcp_cli.commands.prompts import prompts_action_cmd  # shared async helper
from mcp_cli.tools.manager import ToolManager
from .base import InteractiveCommand

log = logging.getLogger(__name__)


class PromptsCommand(InteractiveCommand):
    """Display stored prompt templates found on all servers."""

    def __init__(self) -> None:
        super().__init__(
            name="prompts",
            aliases=["pr"],  # avoid clash with /provider ("p")
            help_text="List prompt templates available on connected MCP servers.",
        )

    # ------------------------------------------------------------------
    async def execute(  # noqa: D401  (simple delegation)
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **ctx: Dict[str, Any],
    ) -> None:
        """
        Delegate to :func:`mcp_cli.commands.prompts.prompts_action_cmd`.
        """

        if tool_manager is None:
            log.debug("PromptsCommand executed without ToolManager - aborting.")
            output.print("[red]Error:[/red] ToolManager not available.")
            return

        # No sub-arguments are supported right now:
        _ = args  # kept for future flags

        await prompts_action_cmd(tool_manager)
