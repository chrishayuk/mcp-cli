# mcp_cli/interactive/commands/base.py
"""Base class for interactive commands."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List

from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)


class InteractiveCommand(ABC):
    """Base class for interactive mode commands.

    All command classes should use 'help' as the attribute name for help text,
    though the constructor parameter is named 'help_text' for clarity.
    """

    name: str
    help: str  # Standard attribute name for help text
    aliases: List[str]

    def __init__(self, name: str, help_text: str = "", aliases: List[str] = None):
        """Initialize command with name, help text, and optional aliases.

        Args:
            name: Command name
            help_text: Help text for the command (stored as self.help)
            aliases: Optional list of command aliases
        """
        self.name = name
        self.help = help_text  # Store as 'help' for consistency
        self.aliases = aliases or []

    @abstractmethod
    async def execute(
        self, args: List[str], tool_manager: ToolManager, **kwargs
    ) -> Any:
        """Execute the command with the given arguments."""
        pass
