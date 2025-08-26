# mcp_cli/chat/commands/__init__.py
"""
Command handling system for the MCP CLI chat interface.
"""

from typing import Dict, List, Any, Callable, Awaitable
import re

# Type for command handlers
CommandHandler = Callable[[List[str], Dict[str, Any]], Awaitable[bool]]

# Global registries
_COMMAND_HANDLERS: Dict[str, CommandHandler] = {}
_COMMAND_COMPLETIONS: Dict[str, List[str]] = {}
_COMMAND_ALIASES: Dict[str, str] = {}


def register_command(
    command: str, handler: CommandHandler, completions: List[str] = None
) -> None:
    """
    Register a command handler.

    Args:
        command: The command (starting with /) to register.
        handler: Async function that handles the command.
        completions: List of completion options for this command.
    """
    if not command.startswith("/"):
        raise ValueError(f"Command {command} must start with /")

    # Register the handler
    _COMMAND_HANDLERS[command] = handler

    # Register completion options if provided
    if completions:
        _COMMAND_COMPLETIONS[command] = completions


def register_alias(alias: str, target: str) -> None:
    """
    Register an alias for an existing command.

    Args:
        alias: The alias command (starting with /)
        target: The target command it points to (starting with /)
    """
    if not alias.startswith("/") or not target.startswith("/"):
        raise ValueError("Both alias and target must start with /")

    if target not in _COMMAND_HANDLERS:
        raise ValueError(f"Cannot create alias to unknown command: {target}")

    _COMMAND_ALIASES[alias] = target

    # Also copy any completions
    if target in _COMMAND_COMPLETIONS:
        _COMMAND_COMPLETIONS[alias] = _COMMAND_COMPLETIONS[target]


async def handle_command(command_text: str, context: Dict[str, Any]) -> bool:
    """
    Handle a command and return whether it was handled.

    Args:
        command_text: The full command text (starting with /).
        context: The context dictionary to pass to the handler.

    Returns:
        bool: True if the command was handled, False otherwise.
    """
    # Split the command and arguments
    parts = command_text.split()
    if not parts:
        return False

    cmd = parts[0].lower()

    # Check if it's an alias and resolve it
    if cmd in _COMMAND_ALIASES:
        cmd = _COMMAND_ALIASES[cmd]
        # Replace the command part with the resolved alias
        parts[0] = cmd

    # Look up the handler
    handler = _COMMAND_HANDLERS.get(cmd)
    if not handler:
        return False

    # Call the handler with args and context
    return await handler(parts, context)


def get_command_completions(partial_text: str) -> List[str]:
    """
    Get command completions for a partial command.

    Args:
        partial_text: The partially typed command.

    Returns:
        List of possible completions.
    """
    completions = []

    # Split into command and arguments
    parts = partial_text.strip().split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    has_arg = len(parts) > 1

    # If no specific argument, suggest commands that start with the partial input
    if not has_arg:
        for command in _COMMAND_HANDLERS:
            if command.startswith(cmd):
                completions.append(command)

        # Also add aliases
        for alias in _COMMAND_ALIASES:
            if alias.startswith(cmd):
                completions.append(alias)

    # If we have an argument, suggest completions for this specific command
    elif cmd in _COMMAND_COMPLETIONS:
        arg_part = parts[1].strip() if len(parts) > 1 else ""
        for completion in _COMMAND_COMPLETIONS[cmd]:
            # Handle parameter placeholders like <filename>
            if completion.startswith("<") and completion.endswith(">"):
                completions.append(f"{cmd} {completion}")
            # Regular completions that match the partial argument
            elif completion.startswith(arg_part):
                completions.append(f"{cmd} {completion}")

    return completions


# Import any built-in command modules here
# This allows them to self-register their commands
import importlib
import pkgutil
import sys
from pathlib import Path


def _import_submodules():
    """Import all submodules to allow them to register their commands."""
    # Get the path of this package
    package_path = Path(__file__).parent

    # Find all modules in this package
    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_path)]):
        if not module_name.startswith("_"):  # Skip private modules
            try:
                importlib.import_module(f"{__package__}.{module_name}")
            except ImportError as e:
                # Log but don't crash
                print(f"Warning: Could not import command module {module_name}: {e}")


# Auto-import all command modules
_import_submodules()

# Explicit imports for critical modules to ensure they're loaded even if auto-discovery fails
try:
    from . import provider
    from . import model
except ImportError as e:
    print(f"Warning: Failed to import critical command module: {e}")
