# src/mcp_cli/commands/registry.py
"""
Unified command registry for all MCP CLI commands.

This is the single source of truth for all commands across all modes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from mcp_cli.commands.base import CommandMode, UnifiedCommand, CommandGroup

logger = logging.getLogger(__name__)


class UnifiedCommandRegistry:
    """
    Central registry for all commands.

    This replaces the separate registries in chat, CLI, and interactive modes.
    """

    _instance: Optional[UnifiedCommandRegistry] = None
    _commands: Dict[str, UnifiedCommand] = {}
    _groups: Dict[str, CommandGroup] = {}
    _initialized: bool = False

    def __new__(cls):
        """Singleton pattern to ensure one registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if not self._initialized:
            self._commands = {}
            self._groups = {}
            self._initialized = True

    def register(self, command: UnifiedCommand, group: Optional[str] = None) -> None:
        """
        Register a command.

        Args:
            command: The command to register.
            group: Optional group name (e.g., 'tools' for 'tools list').
        """
        if group:
            # Register as a subcommand in a group
            if group not in self._groups:
                # Groups must be pre-registered, cannot create them dynamically
                logger.warning(
                    f"Group '{group}' not found for command '{command.name}'"
                )
                return

            self._groups[group].add_subcommand(command)
            logger.debug(f"Registered subcommand: {group} {command.name}")
        else:
            # Register as a top-level command
            self._commands[command.name] = command

            # Also register aliases
            for alias in command.aliases:
                self._commands[alias] = command

            logger.debug(f"Registered command: {command.name}")

    def get(
        self, name: str, mode: Optional[CommandMode] = None
    ) -> Optional[UnifiedCommand]:
        """
        Get a command by name.

        Args:
            name: Command name (can include subcommand like 'tools list').
            mode: Optional mode filter to only return commands that support this mode.

        Returns:
            The command if found and supported in the mode, None otherwise.
        """
        # Handle potential subcommands
        parts = name.split(maxsplit=1)
        command_name = parts[0]

        # Get the command
        command = self._commands.get(command_name)
        if not command:
            return None

        # Check mode support
        if mode and not (command.modes & mode):
            return None

        # If this is a group and we have a subcommand, get the subcommand
        if isinstance(command, CommandGroup) and len(parts) > 1:
            subcommand_name = parts[1]
            subcommand = command.subcommands.get(subcommand_name)
            if subcommand and (not mode or (subcommand.modes & mode)):
                return subcommand
            return None

        return command

    def list_commands(self, mode: Optional[CommandMode] = None) -> List[UnifiedCommand]:
        """
        List all registered commands.

        Args:
            mode: Optional mode filter to only list commands that support this mode.

        Returns:
            List of commands (excluding aliases).
        """
        commands = []
        seen = set()

        for name, command in self._commands.items():
            # Skip aliases (commands will appear multiple times in the dict)
            if command in seen:
                continue

            # Skip hidden commands
            if command.hidden:
                continue

            # Check mode support
            if mode and not (command.modes & mode):
                continue

            commands.append(command)
            seen.add(command)

        return sorted(commands, key=lambda c: c.name)

    def get_command_names(
        self, mode: Optional[CommandMode] = None, include_aliases: bool = False
    ) -> List[str]:
        """
        Get all command names.

        Args:
            mode: Optional mode filter.
            include_aliases: Whether to include command aliases.

        Returns:
            List of command names.
        """
        names = []
        seen_commands = set()

        for name, command in self._commands.items():
            # Check mode support
            if mode and not (command.modes & mode):
                continue

            # Skip hidden commands
            if command.hidden:
                continue

            if include_aliases:
                names.append(name)
            else:
                # Only include primary names
                if command not in seen_commands:
                    names.append(command.name)
                    seen_commands.add(command)

        return sorted(names)

    def clear(self) -> None:
        """Clear all registered commands (useful for testing)."""
        self._commands.clear()
        self._groups.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False


# Global registry instance
registry = UnifiedCommandRegistry()
