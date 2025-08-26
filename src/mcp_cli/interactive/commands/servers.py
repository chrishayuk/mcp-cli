# mcp_cli/interactive/commands/servers.py
"""
Streamlined interactive **servers** command with subcommand support.

Usage Examples:
    servers                     # List all servers with compact view
    servers <name>              # Show detailed info for specific server
    servers <name> enable       # Enable a disabled server
    servers <name> disable      # Disable a server
    servers <name> config       # Show server configuration
    servers <name> tools        # List available tools
    servers <name> test         # Test server connection
    srv                         # Short alias
"""

from __future__ import annotations

import logging
import json
from typing import Any, List, Dict
from pathlib import Path

from chuk_term.ui import output
from chuk_term.ui.prompts import ask, confirm
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from mcp_cli.commands.servers import servers_action_async
from mcp_cli.tools.manager import ToolManager
from .base import InteractiveCommand

log = logging.getLogger(__name__)


async def get_server_details(tm: ToolManager, server_index: int) -> Dict[str, Any]:
    """Get comprehensive details about a specific server."""
    details = {
        "index": server_index,
        "name": f"server-{server_index}",
        "status": "connected",
        "type": "stdio",
        "streamable": False,
        "tools": [],
        "tool_count": 0,
        "capabilities": {},
        "config": {},
        "command": "unknown",
        "args": [],
        "env": {},
    }

    try:
        # Get server info from ToolManager
        if hasattr(tm, "servers") and tm.servers and server_index < len(tm.servers):
            server = tm.servers[server_index]

            if hasattr(server, "name"):
                details["name"] = getattr(server, "name", details["name"])

            if hasattr(server, "transport"):
                details["type"] = getattr(server, "transport", "stdio")
            elif hasattr(server, "connection_type"):
                details["type"] = getattr(server, "connection_type", "stdio")

            if hasattr(server, "supports_streaming"):
                details["streamable"] = getattr(server, "supports_streaming", False)
            elif hasattr(server, "capabilities"):
                caps = getattr(server, "capabilities", {})
                details["streamable"] = caps.get("streaming", False)

            if hasattr(server, "config"):
                config = getattr(server, "config", {})
                details["config"] = config
                details["command"] = config.get("command", "unknown")
                details["args"] = config.get("args", [])
                details["env"] = config.get("env", {})
    except Exception:
        pass

    # Get tools and capabilities
    try:
        if hasattr(tm, "list_tools"):
            all_tools = await tm.list_tools()
            details["tools"] = all_tools
            details["tool_count"] = len(all_tools)
    except Exception:
        pass

    if not details["capabilities"] and details["tool_count"] > 0:
        details["capabilities"]["tools"] = True

    return details


async def display_servers_table(
    servers: List[Dict[str, Any]], show_details: bool = False
) -> None:
    """Display servers in an enhanced table format."""

    table = Table(title="🌐 MCP Servers", header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Server", style="green", width=20)
    table.add_column("Type", width=8)
    table.add_column("Status", width=10)
    table.add_column("Tools", justify="right", width=6)
    table.add_column("Stream", width=8)

    if show_details:
        table.add_column("Capabilities", width=15)
        table.add_column("Command", width=30)

    for i, server in enumerate(servers, 1):
        type_icon = {"stdio": "📝", "sse": "📡", "http": "🌐", "websocket": "🔌"}.get(
            server.get("type", "stdio"), "❓"
        )

        status = server.get("status", "connected")
        if status in ["connected", "ready"]:
            status_display = "[green]● Connected[/green]"
        elif status == "disabled":
            status_display = "[red]● Disabled[/red]"
        else:
            status_display = "[yellow]● Unknown[/yellow]"

        stream_display = "✅ Yes" if server.get("streamable") else "❌ No"
        tool_count = server.get("tool_count", len(server.get("tools", [])))

        row = [
            str(i),
            server["name"],
            f"{type_icon} {server.get('type', 'stdio').upper()}",
            status_display,
            str(tool_count),
            stream_display,
        ]

        if show_details:
            caps = server.get("capabilities", {})
            cap_icons = []
            if caps.get("tools"):
                cap_icons.append("🔧")
            if caps.get("resources"):
                cap_icons.append("📁")
            if caps.get("prompts"):
                cap_icons.append("💬")
            if caps.get("logging"):
                cap_icons.append("📋")
            row.append(" ".join(cap_icons) if cap_icons else "None")

            command = server.get("command", "unknown")
            if len(command) > 25:
                command = command[:22] + "..."
            row.append(command)

        table.add_row(*row)

    output.print(table)


async def display_server_detail(server: Dict[str, Any]) -> None:
    """Display detailed view of a single server."""

    panels = []

    # Basic Info Panel
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Property", style="cyan", width=15)
    info_table.add_column("Value", style="white")

    info_table.add_row("Name", server["name"])
    info_table.add_row("Type", server.get("type", "stdio").upper())
    info_table.add_row("Status", server.get("status", "connected"))
    info_table.add_row("Streamable", "Yes" if server.get("streamable") else "No")
    info_table.add_row("Tools Count", str(server.get("tool_count", 0)))

    panels.append(Panel(info_table, title="📊 Server Information", border_style="blue"))

    # Configuration Panel
    config_data = {
        "command": server.get("command", "unknown"),
        "args": server.get("args", []),
        "env": server.get("env", {}),
    }

    if config_data["command"] != "unknown" or config_data["args"] or config_data["env"]:
        config_json = json.dumps(config_data, indent=2)
        config_syntax = Syntax(config_json, "json", theme="monokai", line_numbers=False)
        panels.append(
            Panel(config_syntax, title="⚙️ Configuration", border_style="green")
        )

    # Capabilities Panel
    caps = server.get("capabilities", {})
    if caps:
        cap_table = Table(show_header=False, box=None)
        cap_table.add_column("Capability", style="yellow", width=20)
        cap_table.add_column("Enabled", style="white")

        for cap, enabled in caps.items():
            if isinstance(enabled, bool):
                cap_table.add_row(cap.title(), "✅" if enabled else "❌")
            elif isinstance(enabled, dict) and enabled:
                cap_table.add_row(cap.title(), "✅ " + ", ".join(enabled.keys()))

        panels.append(Panel(cap_table, title="🎯 Capabilities", border_style="yellow"))

    # Tools Panel
    tools = server.get("tools", [])
    if tools:
        tools_list = []
        for tool in tools[:10]:
            if isinstance(tool, dict):
                tools_list.append(f"• {tool.get('name', 'unknown')}")
            else:
                tools_list.append(f"• {str(tool)}")

        if len(tools) > 10:
            tools_list.append(f"... and {len(tools) - 10} more")

        tools_text = "\n".join(tools_list)
        panels.append(
            Panel(tools_text, title="🔧 Available Tools", border_style="magenta")
        )

    output.rule(f"Server Details: {server['name']}")
    for panel in panels:
        output.print(panel)
        output.print()


class ServersCommand(InteractiveCommand):
    """Enhanced server information display with comprehensive details."""

    def __init__(self) -> None:
        super().__init__(
            name="servers",
            aliases=["srv"],
            help_text=(
                "Streamlined server management with subcommand support.\n\n"
                "Usage:\n"
                "  servers                     - List all servers\n"
                "  servers <name>              - Show server details\n"
                "  servers <name> enable       - Enable server\n"
                "  servers <name> disable      - Disable server\n"
                "  servers <name> config       - Show configuration\n"
                "  servers <name> tools        - List available tools\n"
                "  servers <name> test         - Test connection\n\n"
                "Examples:\n"
                "  servers                     # List all servers\n"
                "  servers sqlite              # Show sqlite details\n"
                "  servers sqlite config       # Show sqlite configuration\n"
                "  servers perplexity disable  # Disable perplexity server"
            ),
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: ToolManager | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Execute the enhanced servers command with full option support.

        Args:
            args: Command line arguments to parse
            tool_manager: ToolManager instance
        """

        if tool_manager is None:
            output.error("ToolManager not available.")
            log.debug("ServersCommand executed without a ToolManager instance.")
            return

        # Parse command line arguments
        parsed_options = self._parse_arguments(args)

        # Handle help request
        if parsed_options.get("help"):
            self._show_help()
            return

        # Handle enable/disable
        if parsed_options.get("enable"):
            server_name = parsed_options["enable"]
            config_path = kwargs.get("config_path", "server_config.json")
            await self._toggle_server_status(config_path, server_name, True)
            return

        if parsed_options.get("disable"):
            server_name = parsed_options["disable"]
            config_path = kwargs.get("config_path", "server_config.json")
            await self._toggle_server_status(config_path, server_name, False)
            return

        # Check for legacy mode
        if parsed_options["format"] != "table" or (
            parsed_options["detailed"] and not parsed_options.get("interactive")
        ):
            # Use legacy servers_action_async for non-interactive modes
            try:
                await servers_action_async(
                    tool_manager,
                    detailed=parsed_options["detailed"],
                    show_capabilities=parsed_options["capabilities"],
                    show_transport=parsed_options["transport"],
                    output_format=parsed_options["format"],
                )
            except Exception as e:
                output.error(f"Failed to display server information: {e}")
                log.error(f"ServersCommand failed: {e}")
            return

        # Interactive mode
        await self._interactive_mode(tool_manager, parsed_options, kwargs)

    async def _interactive_mode(
        self, tool_manager: ToolManager, options: dict, kwargs: dict
    ) -> None:
        """Handle interactive server selection and management."""

        servers = []
        all_server_names = set()
        connected_server_names = set()
        connected_servers = {}

        # Load ALL servers from config file (including disabled ones)
        config_path = "server_config.json"
        config = {}

        try:
            from pathlib import Path
            import json

            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)

                    # Add all servers from config to our list
                    if "mcpServers" in config:
                        for server_name in config["mcpServers"].keys():
                            all_server_names.add(server_name)
        except Exception:
            pass

        try:
            # Get server information from ToolManager (only connected/enabled servers)
            if hasattr(tool_manager, "get_server_info"):
                server_info = await tool_manager.get_server_info()
            else:
                server_info = []

            if not server_info and hasattr(tool_manager, "servers"):
                server_info = []
                for i, srv in enumerate(tool_manager.servers):
                    srv_dict = {
                        "id": i,
                        "name": getattr(srv, "name", f"server-{i}"),
                        "status": getattr(srv, "status", "connected"),
                    }
                    server_info.append(srv_dict)

            # Build map of connected servers
            for i, srv in enumerate(server_info):
                if hasattr(srv, "name"):
                    name = srv.name
                elif isinstance(srv, dict):
                    name = srv.get("name", f"server-{i}")
                else:
                    name = f"server-{i}"
                connected_servers[name] = (i, srv)
                connected_server_names.add(name)
                all_server_names.add(name)

        except Exception as e:
            output.error(f"Failed to get server information: {e}")
            # Don't return - we still want to show disabled servers

        # Get preference manager for disabled status
        from mcp_cli.utils.preferences import get_preference_manager

        pref_manager = get_preference_manager()

        # Process all servers (both connected and from config)
        for server_name in all_server_names:
            server_details = {}

            # Check if this server is connected (in ToolManager)
            if server_name in connected_servers:
                # Server is connected - get details from ToolManager
                srv_index, srv = connected_servers[server_name]
                server_details = await get_server_details(tool_manager, srv_index)

                # Merge with basic info
                if hasattr(srv, "name"):
                    server_details["name"] = srv.name
                elif isinstance(srv, dict):
                    server_details["name"] = srv.get("name", server_name)
                else:
                    server_details["name"] = server_name

                if hasattr(srv, "status"):
                    server_details["status"] = srv.status
                elif isinstance(srv, dict):
                    server_details["status"] = srv.get("status", "connected")
                else:
                    server_details["status"] = "connected"

                server_details["connected"] = True
            else:
                # Server is not connected - create minimal entry
                server_details = {
                    "name": server_name,
                    "status": "disconnected",
                    "type": "stdio",
                    "tools": [],
                    "tool_count": 0,
                    "capabilities": {},
                    "config": {},
                    "command": "unknown",
                    "args": [],
                    "env": {},
                    "enabled": True,
                    "connected": False,
                    "origin": "user",
                    "current": False,
                }

                # Try to get config info for this server
                if "mcpServers" in config:
                    server_config = config["mcpServers"].get(server_name, {})
                    if server_config:
                        server_details["command"] = server_config.get(
                            "command", "unknown"
                        )
                        server_details["args"] = server_config.get("args", [])
                        server_details["env"] = server_config.get("env", {})
                        server_details["config"] = server_config

                        # Determine type from config
                        if "url" in server_config:
                            server_details["type"] = "http"
                        elif "command" in server_config:
                            server_details["type"] = "stdio"

            # Check if server is disabled in preferences
            server_details["enabled"] = not pref_manager.is_server_disabled(server_name)

            servers.append(server_details)

        if not servers:
            # Try fallback
            try:
                if hasattr(tool_manager, "list_tools"):
                    tools = await tool_manager.list_tools()
                    if tools:
                        servers.append(
                            {
                                "name": "default",
                                "status": "connected",
                                "type": "stdio",
                                "streamable": False,
                                "tools": tools,
                                "tool_count": len(tools),
                                "capabilities": {"tools": True},
                                "command": "unknown",
                                "args": [],
                                "env": {},
                            }
                        )
            except Exception:
                pass

        if not servers:
            output.warning("No servers connected")
            output.hint("Connect to a server first")
            return

        # Display table
        await display_servers_table(servers, show_details=options["detailed"])

        if len(servers) >= 1:
            output.print()

            if len(servers) > 1:
                response = ask(
                    f"Enter server number (1-{len(servers)}) for details:",
                    default="",
                    show_default=False,
                )
            else:
                if confirm("Show server details?"):
                    response = "1"
                else:
                    response = ""

            if response and response.isdigit():
                idx = int(response) - 1
                if 0 <= idx < len(servers):
                    output.print()
                    await display_server_detail(servers[idx])

                    # Offer actions
                    output.print()
                    output.rule("Available Actions")
                    output.print("1. View full configuration")
                    output.print("2. Test server connection")
                    output.print("3. Return")

                    action = ask("Select action (1-3):", default="3")

                    if action == "1":
                        config_data = {
                            "command": servers[idx].get("command", "unknown"),
                            "args": servers[idx].get("args", []),
                            "env": servers[idx].get("env", {}),
                        }
                        output.rule("Full Configuration")
                        config_json = json.dumps(config_data, indent=2)
                        syntax = Syntax(config_json, "json", theme="monokai")
                        output.print(syntax)

                    elif action == "2":
                        output.info("Testing server connection...")
                        try:
                            if hasattr(tool_manager, "list_tools"):
                                tools = await tool_manager.list_tools()
                                if tools:
                                    output.success(
                                        f"Server is responding with {len(tools)} tools"
                                    )
                                else:
                                    output.warning(
                                        "Server is responding but has no tools"
                                    )
                            else:
                                output.warning(
                                    "Connection test functionality not available"
                                )
                        except Exception as e:
                            output.error(f"Connection test failed: {e}")
                else:
                    output.error(f"Invalid selection: {response}")

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
            "enable": None,
            "disable": None,
            "interactive": True,
        }

        valid_formats = ["table", "tree", "json"]

        i = 0
        while i < len(args):
            arg = args[i].lower()

            # Help flags
            if arg in ["--help", "-h", "help"]:
                options["help"] = True

            # Enable/Disable flags
            elif arg == "--enable" and i + 1 < len(args):
                options["enable"] = args[i + 1]
                i += 1
            elif arg == "--disable" and i + 1 < len(args):
                options["disable"] = args[i + 1]
                i += 1

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
                        options["interactive"] = False
                        i += 1  # Skip the format value

            # Format shortcuts and flags (--json, --tree)
            elif arg in ["--json", "json"]:
                options["format"] = "json"
                options["interactive"] = False
            elif arg in ["--tree", "tree"]:
                options["format"] = "tree"
                options["interactive"] = False

            # Combined short flags (e.g., -dct)
            elif arg.startswith("-") and len(arg) > 2 and not arg.startswith("--"):
                for char in arg[1:]:
                    if char == "d":
                        options["detailed"] = True
                    elif char == "c":
                        options["capabilities"] = True
                    elif char == "t":
                        options["transport"] = True
                    elif char == "h":
                        options["help"] = True

            i += 1

        # Auto-enable features for detailed view
        if options["detailed"]:
            options["capabilities"] = True
            options["transport"] = True

        return options

    async def _toggle_server_status(
        self, config_path: str, server_name: str, enable: bool
    ) -> bool:
        """Enable or disable a server in the configuration."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                output.error(f"Configuration file not found: {config_path}")
                return False

            with open(config_file, "r") as f:
                config = json.load(f)

            if "mcpServers" not in config:
                config["mcpServers"] = {}

            if server_name not in config["mcpServers"]:
                output.error(f"Server '{server_name}' not found in configuration")
                return False

            if "disabled" not in config["mcpServers"][server_name]:
                config["mcpServers"][server_name]["disabled"] = False

            config["mcpServers"][server_name]["disabled"] = not enable

            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)

            status = "enabled" if enable else "disabled"
            output.success(f"Server '{server_name}' has been {status}")
            output.hint("Restart the session for changes to take effect")
            return True

        except Exception as e:
            output.error(f"Failed to update configuration: {e}")
            return False

    def _show_help(self) -> None:
        """Display comprehensive help information."""
        output.print("[bold cyan]servers[/bold cyan] - Display MCP server information")
        output.print()
        output.print("[bold yellow]Usage:[/bold yellow]")
        output.print("  servers [options]")
        output.print("  srv [options]                    # Short alias")
        output.print()
        output.print("[bold yellow]Options:[/bold yellow]")
        output.print("  --detailed, -d                  Show detailed information")
        output.print("  --capabilities, -c              Include capability information")
        output.print("  --transport, -t                 Include transport details")
        output.print(
            "  --format <fmt>                  Output format: table, tree, json"
        )
        output.print("  --enable <name>                 Enable a disabled server")
        output.print("  --disable <name>                Disable a server")
        output.print("  --help, -h                      Show this help message")
        output.print()
        output.print("[bold yellow]Examples:[/bold yellow]")
        output.print("  servers                         # Interactive selection")
        output.print("  servers --detailed              # Full detailed view")
        output.print("  servers -d                      # Same as --detailed")
        output.print("  servers --format tree           # Tree format display")
        output.print("  servers --enable sqlite         # Enable sqlite server")
        output.print("  servers --disable perplexity    # Disable perplexity server")


# Export command class
__all__ = ["ServersCommand"]
