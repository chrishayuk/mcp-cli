# src/mcp_cli/commands/__init__.py
"""
Unified command system for MCP CLI.

This module provides a single command implementation that works across:
- Chat mode (slash commands)
- CLI mode (typer subcommands)
- Interactive mode (shell commands)
"""

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandGroup,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from mcp_cli.commands.registry import registry, UnifiedCommandRegistry


def register_all_commands() -> None:
    """
    Register all built-in commands with the unified registry.
    
    This should be called once during application startup.
    """
    # Import all command implementations
    from mcp_cli.commands.impl.servers import ServersCommand
    from mcp_cli.commands.impl.help import HelpCommand
    from mcp_cli.commands.impl.exit import ExitCommand
    from mcp_cli.commands.impl.clear import ClearCommand
    from mcp_cli.commands.impl.tools import ToolsCommand
    from mcp_cli.commands.impl.provider import ProviderCommand
    from mcp_cli.commands.impl.model import ModelCommand
    from mcp_cli.commands.impl.ping import PingCommand
    from mcp_cli.commands.impl.theme import ThemeCommand
    from mcp_cli.commands.impl.resources import ResourcesCommand
    from mcp_cli.commands.impl.prompts import PromptsCommand
    from mcp_cli.commands.impl.conversation import ConversationCommand
    from mcp_cli.commands.impl.verbose import VerboseCommand
    
    # Register basic commands
    registry.register(HelpCommand())
    registry.register(ExitCommand())
    registry.register(ClearCommand())
    registry.register(ServersCommand())
    registry.register(PingCommand())
    registry.register(ThemeCommand())
    registry.register(ResourcesCommand())
    registry.register(PromptsCommand())
    
    # Register command groups
    registry.register(ToolsCommand())
    registry.register(ProviderCommand())
    registry.register(ModelCommand())
    
    # Register chat-specific commands
    registry.register(ConversationCommand())
    registry.register(VerboseCommand())
    
    # All commands have been migrated!
    # - tools (with subcommands: list, call, confirm)
    # - provider (with subcommands: list, set, show)
    # - model (with subcommands: list, set, show)
    # - resources
    # - prompts
    # - clear
    # - exit
    # - help
    # - theme
    # - ping
    # - conversation (chat mode only)
    # - verbose (chat mode only)


__all__ = [
    "UnifiedCommand",
    "CommandGroup",
    "CommandMode",
    "CommandParameter",
    "CommandResult",
    "registry",
    "UnifiedCommandRegistry",
    "register_all_commands",
]