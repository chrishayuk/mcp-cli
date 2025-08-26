# mcp_cli/interactive/commands/__init__.py
"""Interactive commands package."""

from .help import HelpCommand
from .exit import ExitCommand
from .clear import ClearCommand
from .servers import ServersCommand
from .tools import ToolsCommand
from .resources import ResourcesCommand
from .prompts import PromptsCommand
from .ping import PingCommand
from .model import ModelCommand
from .provider import ProviderCommand
from .theme import ThemeCommand

# Export for convenience
__all__ = [
    "HelpCommand",
    "ExitCommand",
    "ClearCommand",
    "ServersCommand",
    "ToolsCommand",
    "ResourcesCommand",
    "PromptsCommand",
    "PingCommand",
    "ModelCommand",
    "ProviderCommand",
    "ThemeCommand",
]


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
    from mcp_cli.interactive.commands.model import ModelCommand
    from mcp_cli.interactive.commands.provider import ProviderCommand
    from mcp_cli.interactive.commands.theme import ThemeCommand

    reg = InteractiveCommandRegistry
    reg.register(HelpCommand())
    reg.register(ExitCommand())
    reg.register(ClearCommand())
    reg.register(ServersCommand())
    reg.register(ToolsCommand())
    reg.register(ResourcesCommand())
    reg.register(PromptsCommand())
    reg.register(PingCommand())
    reg.register(ModelCommand())
    reg.register(ProviderCommand())
    reg.register(ThemeCommand())
