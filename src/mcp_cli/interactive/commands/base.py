# mcp_cli/interactive/commands/base.py
"""Base class for interactive commands."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Awaitable

from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)


class InteractiveCommand(ABC):
    """Base class for interactive mode commands."""
    
    name: str
    help: str
    aliases: List[str]
    
    def __init__(self, name: str, help_text: str = "", aliases: List[str] = None):
        self.name = name
        self.help = help_text
        self.aliases = aliases or []
    
    @abstractmethod
    async def execute(self, args: List[str], tool_manager: ToolManager, **kwargs) -> Any:
        """Execute the command with the given arguments."""
        pass