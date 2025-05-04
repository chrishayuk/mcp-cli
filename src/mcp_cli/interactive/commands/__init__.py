# mcp_cli/interactive/commands/__init__.py
"""Interactive commands package."""
from .help import HelpCommand
from .exit import ExitCommand
from .clear import ClearCommand
from .servers import ServersCommand
from .tools import ToolsCommand
from .resources import ResourcesCommand
from .prompts import PromptsCommand

# Export for convenience
__all__ = [
    "HelpCommand",
    "ExitCommand",
    "ClearCommand",
    "ServersCommand",
    "ToolsCommand",
    "ResourcesCommand",
    "PromptsCommand",
]

def register_all_commands():
    """Register every interactive command into the central registry."""
    from mcp_cli.interactive.registry import InteractiveCommandRegistry

    # Instantiate and register each command
    InteractiveCommandRegistry.register(HelpCommand())
    InteractiveCommandRegistry.register(ExitCommand())
    InteractiveCommandRegistry.register(ClearCommand())
    InteractiveCommandRegistry.register(ServersCommand())
    InteractiveCommandRegistry.register(ToolsCommand())
    InteractiveCommandRegistry.register(ResourcesCommand())
    InteractiveCommandRegistry.register(PromptsCommand())
