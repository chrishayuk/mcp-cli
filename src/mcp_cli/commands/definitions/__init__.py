# src/mcp_cli/commands/definitions/__init__.py
"""
Command definitions for MCP CLI.

This module contains all command class definitions that specify
the interface and parameters for each command.
"""

from .clear import ClearCommand
from .conversation import ConversationCommand
from .exit import ExitCommand
from .help import HelpCommand
from .interrupt import InterruptCommand
from .models import ModelCommand
from .ping import PingCommand
from .prompts import PromptsCommand
from .providers import ProviderCommand
from .resources import ResourcesCommand
from .servers import ServersCommand
from .theme import ThemeCommand
from .token import TokenCommand
from .tool_history import ToolHistoryCommand
from .tools import ToolsCommand
from .verbose import VerboseCommand

__all__ = [
    "ClearCommand",
    "ConversationCommand",
    "ExitCommand",
    "HelpCommand",
    "InterruptCommand",
    "ModelCommand",
    "PingCommand",
    "PromptsCommand",
    "ProviderCommand",
    "ResourcesCommand",
    "ServersCommand",
    "ThemeCommand",
    "TokenCommand",
    "ToolHistoryCommand",
    "ToolsCommand",
    "VerboseCommand",
]
