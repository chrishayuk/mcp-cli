# mcp_cli/cli/commands/__init__.py
def register_all_commands() -> None:
    """
    Instantiate and register every CLI command into the registry.
    """
    # Delay import to avoid circular dependencies
    from mcp_cli.cli.registry import CommandRegistry

    # Core "mode" commands
    from mcp_cli.cli.commands.interactive import InteractiveCommand
    from mcp_cli.cli.commands.chat import ChatCommand
    from mcp_cli.cli.commands.cmd import CmdCommand
    from mcp_cli.cli.commands.ping import PingCommand
    from mcp_cli.cli.commands.provider import ProviderCommand

    # Sub-app commands
    from mcp_cli.cli.commands.tools import ToolsListCommand
    from mcp_cli.cli.commands.tools_call import ToolsCallCommand
    from mcp_cli.cli.commands.prompts import PromptsListCommand
    from mcp_cli.cli.commands.resources import ResourcesListCommand
    from mcp_cli.cli.commands.servers import ServersListCommand  # ← NEW

    # Register everything in the central registry
    CommandRegistry.register(InteractiveCommand())
    CommandRegistry.register(ChatCommand())
    CommandRegistry.register(CmdCommand())
    CommandRegistry.register(PingCommand())
    CommandRegistry.register(ProviderCommand())

    CommandRegistry.register(ToolsListCommand())
    CommandRegistry.register(ToolsCallCommand())
    CommandRegistry.register(PromptsListCommand())
    CommandRegistry.register(ResourcesListCommand())
    CommandRegistry.register(ServersListCommand())  # ← NEW
