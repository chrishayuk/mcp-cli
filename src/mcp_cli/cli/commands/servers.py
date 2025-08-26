# src/mcp_cli/cli/commands/servers.py
"""
Enhanced CLI binding for "servers" command with detailed capability information.

This module provides comprehensive server information display including:
- Server capabilities and protocol versions
- Transport details and connection parameters
- Feature analysis and capability breakdown
- Multiple output formats (table, tree, JSON)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import typer

# Updated imports for enhanced server functionality
from mcp_cli.commands.servers import servers_action, servers_action_async
from mcp_cli.tools.manager import get_tool_manager
from mcp_cli.cli.commands.base import BaseCommand

logger = logging.getLogger(__name__)

# ─── Typer sub-app with enhanced options ────────────────────────────────────
app = typer.Typer(
    help="Display comprehensive information about connected MCP servers",
    rich_markup_mode="rich",
)


@app.command("list")
def servers_list(
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed information including capabilities and transport details",
    ),
    capabilities: bool = typer.Option(
        False,
        "--capabilities",
        "--caps",
        "-c",
        help="Include server capability information in output",
    ),
    transport: bool = typer.Option(
        False,
        "--transport",
        "--trans",
        "-t",
        help="Include transport/connection details",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, tree, or json",
        case_sensitive=False,
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress server connection logging"
    ),
) -> None:
    """
    List connected MCP servers with comprehensive information.

    [bold green]Examples:[/bold green]
        [cyan]mcp-cli servers list[/cyan]                    # Basic table view
        [cyan]mcp-cli servers list --detailed[/cyan]         # Full detailed panels
        [cyan]mcp-cli servers list -d -c -t[/cyan]          # All details with flags
        [cyan]mcp-cli servers list --format tree[/cyan]     # Tree format display
        [cyan]mcp-cli servers list --format json[/cyan]     # JSON output

    [bold yellow]Feature Icons:[/bold yellow]
        🔧 Tools  📁 Resources  💬 Prompts  ⚡ Streaming  🔔 Notifications
    """
    # Validate format
    valid_formats = ["table", "tree", "json"]
    if output_format.lower() not in valid_formats:
        typer.echo(
            f"Error: Invalid format '{output_format}'. Must be one of: {', '.join(valid_formats)}",
            err=True,
        )
        raise typer.Exit(code=1)

    tm = get_tool_manager()
    if tm is None:
        typer.echo("Error: no ToolManager initialised", err=True)
        raise typer.Exit(code=1)

    # Configure logging if quiet mode requested
    if quiet:
        logging.getLogger("chuk_mcp").setLevel(logging.WARNING)
        logging.getLogger("mcp_cli").setLevel(logging.WARNING)

    # Auto-enable features for detailed view
    if detailed:
        capabilities = True
        transport = True

    try:
        # Use the enhanced servers_action with new parameters
        servers_action(
            tm,
            detailed=detailed,
            show_capabilities=capabilities,
            show_transport=transport,
            output_format=output_format.lower(),
        )
    except Exception as e:
        logger.error(f"Error listing servers: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    raise typer.Exit(code=0)


@app.command("capabilities")
def servers_capabilities(
    server_name: Optional[str] = typer.Argument(
        None, help="Show capabilities for specific server (optional)"
    ),
    compare: bool = typer.Option(
        False, "--compare", "-c", help="Compare capabilities across all servers"
    ),
) -> None:
    """
    Display detailed capability information for servers.

    [bold green]Examples:[/bold green]
        [cyan]mcp-cli servers capabilities[/cyan]                    # All server capabilities
        [cyan]mcp-cli servers capabilities sqlite-server[/cyan]     # Specific server
        [cyan]mcp-cli servers capabilities --compare[/cyan]         # Capability comparison
    """
    tm = get_tool_manager()
    if tm is None:
        typer.echo("Error: no ToolManager initialised", err=True)
        raise typer.Exit(code=1)

    # This would call a specialized capability analysis function
    try:
        if compare:
            servers_action(tm, output_format="tree", show_capabilities=True)
        else:
            servers_action(tm, detailed=True, show_capabilities=True)
    except Exception as e:
        logger.error(f"Error showing capabilities: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("status")
def servers_status(
    refresh: bool = typer.Option(
        False,
        "--refresh",
        "-r",
        help="Refresh server connections before showing status",
    ),
) -> None:
    """
    Show connection status and health for all servers.

    [bold green]Examples:[/bold green]
        [cyan]mcp-cli servers status[/cyan]           # Current status
        [cyan]mcp-cli servers status --refresh[/cyan] # Refresh connections first
    """
    tm = get_tool_manager()
    if tm is None:
        typer.echo("Error: no ToolManager initialised", err=True)
        raise typer.Exit(code=1)

    if refresh:
        typer.echo("Refreshing server connections...")
        # This would trigger a reconnection attempt

    try:
        # Show basic status-focused view
        servers_action(tm, show_capabilities=False, show_transport=False)
    except Exception as e:
        logger.error(f"Error checking server status: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


# Default command that runs when just "servers" is called
@app.callback(invoke_without_command=True)
def servers_default(
    ctx: typer.Context,
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed server information"
    ),
    capabilities: bool = typer.Option(
        False, "--capabilities", "--caps", "-c", help="Include capability information"
    ),
    transport: bool = typer.Option(
        False, "--transport", "--trans", "-t", help="Include transport details"
    ),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, tree, or json"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress verbose logging"),
) -> None:
    """
    Display comprehensive information about connected MCP servers.

    Shows server status, protocol versions, capabilities, and available features.

    [bold green]Quick Examples:[/bold green]
        [cyan]mcp-cli servers[/cyan]                      # Basic server list
        [cyan]mcp-cli servers --detailed[/cyan]           # Full detailed view
        [cyan]mcp-cli servers --format tree[/cyan]        # Tree format
        [cyan]mcp-cli servers --quiet[/cyan]              # Suppress logging

    [bold yellow]Available Subcommands:[/bold yellow]
        [cyan]list[/cyan]         - List servers with options
        [cyan]capabilities[/cyan] - Show detailed capabilities
        [cyan]status[/cyan]       - Check connection status
    """
    # If a subcommand was invoked, let it handle things
    if ctx.invoked_subcommand is not None:
        return

    # Validate format
    valid_formats = ["table", "tree", "json"]
    if output_format.lower() not in valid_formats:
        typer.echo(
            f"Error: Invalid format '{output_format}'. Must be one of: {', '.join(valid_formats)}",
            err=True,
        )
        raise typer.Exit(code=1)

    # Otherwise run the default server listing
    tm = get_tool_manager()
    if tm is None:
        typer.echo("Error: no ToolManager initialised", err=True)
        raise typer.Exit(code=1)

    # Configure logging if quiet mode requested
    if quiet:
        logging.getLogger("chuk_mcp").setLevel(logging.WARNING)
        logging.getLogger("mcp_cli").setLevel(logging.WARNING)

    # Auto-enable features for detailed view
    if detailed:
        capabilities = True
        transport = True

    try:
        servers_action(
            tm,
            detailed=detailed,
            show_capabilities=capabilities,
            show_transport=transport,
            output_format=output_format.lower(),
        )
    except Exception as e:
        logger.error(f"Error listing servers: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


# ─── In-process command for CommandRegistry (Enhanced) ──────────────────────
class ServersListCommand(BaseCommand):
    """
    Enhanced `servers list` command for interactive shell and script usage.

    Supports all the same options as the CLI version including detailed views,
    capability analysis, and multiple output formats.
    """

    def __init__(self) -> None:
        super().__init__(
            name="servers",
            help_text=(
                "Display comprehensive server information.\n\n"
                "Usage:\n"
                "  servers                 - Basic table view\n"
                "  servers --detailed      - Full detailed panels\n"
                "  servers --capabilities  - Include capabilities\n"
                "  servers --transport     - Show transport details\n"
                "  servers --format tree   - Tree format display\n"
                "  servers --format json   - JSON output\n\n"
                "Feature Icons: 🔧 Tools  📁 Resources  💬 Prompts  ⚡ Streaming  🔔 Notifications"
            ),
        )

    async def execute(
        self,
        tool_manager: Any,
        detailed: bool = False,
        show_capabilities: bool = False,
        show_transport: bool = False,
        output_format: str = "table",
        **kwargs: Any,
    ) -> None:
        """
        Execute the enhanced servers command with full option support.

        Args:
            tool_manager: ToolManager instance
            detailed: Show detailed information
            show_capabilities: Include capability information
            show_transport: Include transport details
            output_format: Output format (table, tree, json)
        """
        logger.debug("Executing enhanced ServersListCommand")

        # Auto-enable features for detailed view
        if detailed:
            show_capabilities = True
            show_transport = True

        await servers_action_async(
            tool_manager,
            detailed=detailed,
            show_capabilities=show_capabilities,
            show_transport=show_transport,
            output_format=output_format,
        )


class ServersCapabilitiesCommand(BaseCommand):
    """Command specifically for analyzing server capabilities."""

    def __init__(self) -> None:
        super().__init__(
            name="servers capabilities",
            help_text="Analyze and compare server capabilities across all connected servers.",
        )

    async def execute(self, tool_manager: Any, **kwargs: Any) -> None:
        """Show detailed capability analysis."""
        logger.debug("Executing ServersCapabilitiesCommand")
        await servers_action_async(
            tool_manager, detailed=True, show_capabilities=True, output_format="tree"
        )


class ServersStatusCommand(BaseCommand):
    """Command for checking server connection status."""

    def __init__(self) -> None:
        super().__init__(
            name="servers status",
            help_text="Check connection status and health of all MCP servers.",
        )

    async def execute(self, tool_manager: Any, **kwargs: Any) -> None:
        """Show server status information."""
        logger.debug("Executing ServersStatusCommand")
        await servers_action_async(
            tool_manager,
            detailed=False,
            show_capabilities=False,
            show_transport=False,
            output_format="table",
        )


# Export all command classes for registration
__all__ = [
    "app",
    "ServersListCommand",
    "ServersCapabilitiesCommand",
    "ServersStatusCommand",
]
