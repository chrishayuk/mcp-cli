# mcp_cli/interactive/commands/servers.py
"""
Enhanced interactive **servers** command with comprehensive server information display.

This command provides detailed information about connected MCP servers including:
- Server capabilities and protocol versions
- Transport details and connection parameters
- Feature analysis and capability breakdown
- Multiple output formats with rich formatting

Usage Examples:
    servers                     # Basic table with feature icons
    servers --detailed          # Full detailed panels
    servers --capabilities      # Include capability information
    servers --transport         # Show transport details
    servers --format tree       # Tree format display
    servers --format json       # JSON output
    srv -d                      # Short alias with detailed flag
"""
from __future__ import annotations

import logging
from typing import Any, List

from mcp_cli.utils.rich_helpers import get_console
from mcp_cli.commands.servers import servers_action_async
from mcp_cli.tools.manager import ToolManager
from .base import InteractiveCommand

log = logging.getLogger(__name__)


class ServersCommand(InteractiveCommand):
    """Enhanced server information display with comprehensive details."""

    def __init__(self) -> None:
        super().__init__(
            name="servers",
            aliases=["srv"],
            help_text=(
                "Display comprehensive information about connected MCP servers.\n\n"
                "Usage:\n"
                "  servers                     - Basic table with feature icons\n"
                "  servers --detailed          - Full detailed view with panels\n"
                "  servers --capabilities      - Include capability information\n"
                "  servers --transport         - Show transport/connection details\n"
                "  servers --format tree       - Tree format display\n"
                "  servers --format json       - Raw JSON output\n"
                "  servers --quiet             - Suppress verbose logging\n\n"
                "Flags can be combined and shortened:\n"
                "  servers -d                  - Short for --detailed\n"
                "  servers -c                  - Short for --capabilities\n"
                "  servers -t                  - Short for --transport\n"
                "  servers -d -c -t            - All details combined\n\n"
                "Available formats: table (default), tree, json\n\n"
                "Feature Icons: 🔧 Tools  📁 Resources  💬 Prompts  ⚡ Streaming  🔔 Notifications"
            ),
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **_: Any,
    ) -> None:
        """
        Execute the enhanced servers command with full option support.
        
        Args:
            args: Command line arguments to parse
            tool_manager: ToolManager instance
        """
        console = get_console()

        if tool_manager is None:
            console.print("[red]Error:[/red] ToolManager not available.")
            log.debug("ServersCommand executed without a ToolManager instance.")
            return

        # Parse command line arguments
        parsed_options = self._parse_arguments(args)
        
        # Handle help request
        if parsed_options.get("help"):
            self._show_help(console)
            return
        
        # Handle invalid format
        if parsed_options.get("invalid_format"):
            console.print(f"[red]Error:[/red] Invalid format '{parsed_options['invalid_format']}'. Valid formats: table, tree, json")
            return

        try:
            await servers_action_async(
                tool_manager,
                detailed=parsed_options["detailed"],
                show_capabilities=parsed_options["capabilities"],
                show_transport=parsed_options["transport"],
                output_format=parsed_options["format"]
            )
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to display server information: {e}")
            log.error(f"ServersCommand failed: {e}")

    def _parse_arguments(self, args: List[str]) -> dict:
        """
        Parse command line arguments and return options dictionary.
        
        Args:
            args: List of command arguments
            
        Returns:
            Dictionary with parsed options
        """
        options = {
            "detailed": False,
            "capabilities": False,
            "transport": False,
            "format": "table",
            "quiet": False,
            "help": False,
            "invalid_format": None
        }
        
        valid_formats = ["table", "tree", "json"]
        
        i = 0
        while i < len(args):
            arg = args[i].lower()
            
            # Help flags
            if arg in ["--help", "-h", "help"]:
                options["help"] = True
                
            # Detail flags
            elif arg in ["--detailed", "-d", "--detail"]:
                options["detailed"] = True
                
            # Capability flags
            elif arg in ["--capabilities", "--caps", "-c"]:
                options["capabilities"] = True
                
            # Transport flags
            elif arg in ["--transport", "--trans", "-t"]:
                options["transport"] = True
                
            # Format flag with value
            elif arg in ["--format", "-f"]:
                if i + 1 < len(args):
                    format_value = args[i + 1].lower()
                    if format_value in valid_formats:
                        options["format"] = format_value
                        i += 1  # Skip the format value
                    else:
                        options["invalid_format"] = args[i + 1]
                        return options
                else:
                    # Format flag without value - show help
                    options["help"] = True
                    
            # Quiet flag
            elif arg in ["--quiet", "-q"]:
                options["quiet"] = True
                
            # Format shortcuts
            elif arg == "tree":
                options["format"] = "tree"
            elif arg == "json":
                options["format"] = "json"
            elif arg == "table":
                options["format"] = "table"
                
            # Combined short flags (e.g., -dct)
            elif arg.startswith("-") and len(arg) > 2 and not arg.startswith("--"):
                for char in arg[1:]:
                    if char == "d":
                        options["detailed"] = True
                    elif char == "c":
                        options["capabilities"] = True
                    elif char == "t":
                        options["transport"] = True
                    elif char == "q":
                        options["quiet"] = True
                    elif char == "h":
                        options["help"] = True
            
            i += 1
        
        # Auto-enable features for detailed view
        if options["detailed"]:
            options["capabilities"] = True
            options["transport"] = True
            
        return options

    def _show_help(self, console) -> None:
        """Display comprehensive help information."""
        console.print("[bold cyan]servers[/bold cyan] - Display MCP server information")
        console.print()
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print("  servers [options]")
        console.print("  srv [options]                    # Short alias")
        console.print()
        console.print("[bold yellow]Options:[/bold yellow]")
        console.print("  --detailed, -d                  Show detailed information with panels")
        console.print("  --capabilities, --caps, -c      Include server capability information")
        console.print("  --transport, --trans, -t        Include transport/connection details") 
        console.print("  --format <fmt>, -f <fmt>        Output format: table, tree, json")
        console.print("  --quiet, -q                     Suppress verbose logging")
        console.print("  --help, -h                      Show this help message")
        console.print()
        console.print("[bold yellow]Format Options:[/bold yellow]")
        console.print("  table                           Compact table view (default)")
        console.print("  tree                            Hierarchical tree display")
        console.print("  json                            Raw JSON output")
        console.print()
        console.print("[bold yellow]Examples:[/bold yellow]")
        console.print("  servers                         # Basic table with feature icons")
        console.print("  servers --detailed              # Full detailed panels")
        console.print("  servers -d                      # Same as --detailed") 
        console.print("  servers --capabilities          # Include capability info")
        console.print("  servers -c                      # Short for --capabilities")
        console.print("  servers --transport             # Show connection details")
        console.print("  servers -t                      # Short for --transport")
        console.print("  servers -dct                    # All details combined")
        console.print("  servers --format tree           # Tree format display")
        console.print("  servers -f json                 # JSON output")
        console.print("  servers tree                    # Format shortcut")
        console.print("  servers json --quiet            # JSON with reduced logging")
        console.print()
        console.print("[bold yellow]Feature Icons:[/bold yellow]")
        console.print("  🔧 Tools      📁 Resources    💬 Prompts")
        console.print("  ⚡ Streaming  🔔 Notifications")


class ServersCapabilitiesCommand(InteractiveCommand):
    """Specialized command for analyzing server capabilities."""

    def __init__(self) -> None:
        super().__init__(
            name="servers capabilities",
            aliases=["srv caps", "capabilities"],
            help_text=(
                "Analyze and display detailed server capability information.\n\n"
                "This command provides in-depth analysis of what each server supports,\n"
                "including tools, resources, prompts, streaming, and notifications.\n\n"
                "Usage:\n"
                "  servers capabilities            - Show all server capabilities\n"
                "  srv caps                        - Short alias\n"
                "  capabilities                    - Direct command"
            ),
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **_: Any,
    ) -> None:
        """Show detailed capability analysis for all servers."""
        console = get_console()

        if tool_manager is None:
            console.print("[red]Error:[/red] ToolManager not available.")
            log.debug("ServersCapabilitiesCommand executed without ToolManager.")
            return

        try:
            # Use tree format with detailed capabilities for best analysis view
            await servers_action_async(
                tool_manager,
                detailed=True,
                show_capabilities=True,
                show_transport=False,
                output_format="tree"
            )
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to analyze capabilities: {e}")
            log.error(f"ServersCapabilitiesCommand failed: {e}")


class ServersStatusCommand(InteractiveCommand):
    """Specialized command for checking server connection status."""

    def __init__(self) -> None:
        super().__init__(
            name="servers status",
            aliases=["srv status", "status"],
            help_text=(
                "Check connection status and health of all MCP servers.\n\n"
                "Shows current connection state, protocol versions, and basic\n"
                "server information without detailed capability analysis.\n\n"
                "Usage:\n"
                "  servers status                  - Check server status\n"
                "  srv status                      - Short alias\n"
                "  status                          - Direct command"
            ),
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **_: Any,
    ) -> None:
        """Show server connection status and health."""
        console = get_console()

        if tool_manager is None:
            console.print("[red]Error:[/red] ToolManager not available.")
            log.debug("ServersStatusCommand executed without ToolManager.")
            return

        try:
            # Use basic table format focused on status information
            await servers_action_async(
                tool_manager,
                detailed=False,
                show_capabilities=False,
                show_transport=False,
                output_format="table"
            )
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to check server status: {e}")
            log.error(f"ServersStatusCommand failed: {e}")


# Export all command classes
__all__ = [
    "ServersCommand",
    "ServersCapabilitiesCommand", 
    "ServersStatusCommand"
]