# mcp_cli/interactive/commands/help.py
"""
Interactive “help” command - shows global help or details for a single command.
"""
from __future__ import annotations

from typing import Any, List, Optional

from rich.console import Console

from .base import InteractiveCommand
from mcp_cli.commands.help import help_action


class HelpCommand(InteractiveCommand):
    """Display available commands or detailed help for one command."""

    def __init__(self) -> None:
        super().__init__(
            name="help",
            aliases=["?", "h"],
            help_text="Display available commands or help for a specific command.",
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: Any = None,  # unused but kept for interface parity
        **_: Any,
    ) -> None:
        console = Console()
        cmd_name: Optional[str] = args[0] if args else None
        # new signature: first arg is *command_name*, console is keyword-only
        help_action(cmd_name, console=console)
