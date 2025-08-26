# src/mcp_cli/interactive/registry.py
"""Registry for interactive commands."""

from __future__ import annotations
import logging
from typing import Dict, Optional

# commands
from mcp_cli.interactive.commands.base import InteractiveCommand

# logger
logger = logging.getLogger(__name__)


class InteractiveCommandRegistry:
    """Registry for interactive commands."""

    _commands: Dict[str, InteractiveCommand] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(cls, command: InteractiveCommand) -> None:
        """Register a command under its name and any aliases."""
        cls._commands[command.name] = command
        for alias in command.aliases:
            cls._aliases[alias] = command.name

    @classmethod
    def get_command(cls, name: str) -> Optional[InteractiveCommand]:
        """Retrieve a command by name or alias."""
        # resolve alias
        if name in cls._aliases:
            name = cls._aliases[name]
        return cls._commands.get(name)

    @classmethod
    def get_all_commands(cls) -> Dict[str, InteractiveCommand]:
        """Return the mapping of all registered commands."""
        if not isinstance(cls._commands, dict):
            # print("[DEBUG] InteractiveCommandRegistry._commands polluted! Type:", type(cls._commands))
            cls._commands = {}
        return cls._commands


def register_all_commands() -> None:
    """
    Register every interactive command in the central registry.
    """
    from mcp_cli.interactive.registry import InteractiveCommandRegistry
    from mcp_cli.interactive.commands.help import HelpCommand
    from mcp_cli.interactive.commands.exit import ExitCommand
    from mcp_cli.interactive.commands.clear import ClearCommand
    from mcp_cli.interactive.commands.servers import ServersCommand
    from mcp_cli.interactive.commands.tools import ToolsCommand
    from mcp_cli.interactive.commands.resources import ResourcesCommand
    from mcp_cli.interactive.commands.prompts import PromptsCommand
    from mcp_cli.interactive.commands.ping import PingCommand

    reg = InteractiveCommandRegistry
    reg.register(HelpCommand())
    reg.register(ExitCommand())
    reg.register(ClearCommand())
    reg.register(ServersCommand())
    reg.register(ToolsCommand())
    reg.register(ResourcesCommand())
    reg.register(PromptsCommand())
    reg.register(PingCommand())
