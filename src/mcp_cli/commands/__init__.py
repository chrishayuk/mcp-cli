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
from mcp_cli.commands.decorators import validate_params, handle_errors
from mcp_cli.commands.exceptions import (
    CommandError,
    InvalidParameterError,
    CommandExecutionError,
    CommandNotFoundError,
    ValidationError,
)
from mcp_cli.commands import utils as command_utils  # noqa: F401
from mcp_cli.commands.models import (
    # Server models
    ServerActionParams,
    ServerStatusInfo,
    ServerPerformanceInfo,
    # Model models
    ModelActionParams,
    ModelInfo,
    # Provider models
    ProviderActionParams,
    ProviderInfo,
    # Token models
    TokenListParams,
    TokenSetParams,
    TokenDeleteParams,
    TokenClearParams,
    TokenProviderParams,
    # Tool models
    ToolActionParams,
    ToolCallParams,
    # Resource models
    ResourceActionParams,
    # Prompt models
    PromptActionParams,
    # Theme models
    ThemeActionParams,
    ThemeInfo,
    # Conversation models
    ConversationActionParams,
    ConversationInfo,
    # Response models
    ServerInfoResponse,
    ResourceInfoResponse,
    PromptInfoResponse,
    ToolInfoResponse,
)

__all__ = [
    # Base classes
    "UnifiedCommand",
    "CommandGroup",
    "CommandMode",
    "CommandParameter",
    "CommandResult",
    # Registry
    "registry",
    "UnifiedCommandRegistry",
    # Decorators
    "validate_params",
    "handle_errors",
    # Exceptions
    "CommandError",
    "InvalidParameterError",
    "CommandExecutionError",
    "CommandNotFoundError",
    "ValidationError",
    # Utils
    "command_utils",
    # Response models
    "ServerInfoResponse",
    "ResourceInfoResponse",
    "PromptInfoResponse",
    "ToolInfoResponse",
    # Functions
    "register_all_commands",
]


def register_all_commands() -> None:
    """
    Register all built-in commands with the unified registry.

    This should be called once during application startup.
    """
    # Import all command implementations
    from mcp_cli.commands.definitions.servers import ServersCommand
    from mcp_cli.commands.definitions.server_singular import ServerSingularCommand
    from mcp_cli.commands.definitions.help import HelpCommand
    from mcp_cli.commands.definitions.exit import ExitCommand
    from mcp_cli.commands.definitions.clear import ClearCommand
    from mcp_cli.commands.definitions.tools import ToolsCommand
    from mcp_cli.commands.definitions.providers import ProviderCommand
    from mcp_cli.commands.definitions.provider_singular import ProviderSingularCommand
    from mcp_cli.commands.definitions.models import ModelCommand
    from mcp_cli.commands.definitions.ping import PingCommand
    from mcp_cli.commands.definitions.theme_singular import ThemeSingularCommand
    from mcp_cli.commands.definitions.themes_plural import ThemesPluralCommand
    from mcp_cli.commands.definitions.resources import ResourcesCommand
    from mcp_cli.commands.definitions.prompts import PromptsCommand
    from mcp_cli.commands.definitions.conversation import ConversationCommand
    from mcp_cli.commands.definitions.verbose import VerboseCommand
    from mcp_cli.commands.definitions.interrupt import InterruptCommand
    from mcp_cli.commands.definitions.tool_history import ToolHistoryCommand
    from mcp_cli.commands.definitions.execute_tool import ExecuteToolCommand
    from mcp_cli.commands.definitions.token import TokenCommand

    # Register basic commands
    registry.register(HelpCommand())
    registry.register(ExitCommand())
    registry.register(ClearCommand())
    registry.register(TokenCommand())

    # Register server commands (singular and plural)
    registry.register(ServerSingularCommand())  # /server <name> - show details
    registry.register(ServersCommand())  # /servers - list all

    registry.register(PingCommand())
    registry.register(ResourcesCommand())
    registry.register(PromptsCommand())

    # Register theme commands (singular and plural)
    registry.register(ThemeSingularCommand())  # /theme - show current
    registry.register(ThemesPluralCommand())  # /themes - list all
    # Note: Keep old ThemeCommand for backward compatibility if needed

    # Register provider commands (singular and plural)
    registry.register(ProviderSingularCommand())  # /provider - show current
    registry.register(ProviderCommand())  # /providers - list all

    # Register command groups
    registry.register(ToolsCommand())
    registry.register(ModelCommand())

    # Register chat-specific commands
    registry.register(ConversationCommand())
    registry.register(VerboseCommand())
    registry.register(InterruptCommand())
    registry.register(ToolHistoryCommand())

    # Register tool execution command for interactive mode
    registry.register(ExecuteToolCommand())

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
    # Base classes
    "UnifiedCommand",
    "CommandGroup",
    "CommandMode",
    "CommandParameter",
    "CommandResult",
    # Registry
    "registry",
    "UnifiedCommandRegistry",
    "register_all_commands",
    # Decorators
    "validate_params",
    "handle_errors",
    # Server models
    "ServerActionParams",
    "ServerStatusInfo",
    "ServerPerformanceInfo",
    # Model models
    "ModelActionParams",
    "ModelInfo",
    # Provider models
    "ProviderActionParams",
    "ProviderInfo",
    # Token models
    "TokenListParams",
    "TokenSetParams",
    "TokenDeleteParams",
    "TokenClearParams",
    "TokenProviderParams",
    # Tool models
    "ToolActionParams",
    "ToolCallParams",
    # Resource models
    "ResourceActionParams",
    # Prompt models
    "PromptActionParams",
    # Theme models
    "ThemeActionParams",
    "ThemeInfo",
    # Conversation models
    "ConversationActionParams",
    "ConversationInfo",
]
