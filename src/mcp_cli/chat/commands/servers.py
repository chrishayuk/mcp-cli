from __future__ import annotations

"""
Enhanced /servers command with detailed capability and protocol information.

Usage Examples
--------------
/servers                    - Interactive server selection with details
/servers --detailed         - Full detailed view with all information
/servers --select           - Interactive selection mode (default)
/servers --enable <name>    - Enable a disabled server
/servers --disable <name>   - Disable a server
/srv -d                     - Short alias with detailed flag
"""

from typing import Any, Dict, List, Optional
import json
import time
import asyncio
from pathlib import Path

from chuk_term.ui import output
from chuk_term.ui.prompts import confirm
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command
from mcp_cli.utils.preferences import get_preference_manager


async def get_server_diagnostics(
    tm: ToolManager, server_index: int, server_name: str
) -> Dict[str, Any]:
    """Get diagnostic information about a server (like mcp_server_diagnostic.py)."""
    diagnostics = {
        "ping_time": None,
        "tool_list_time": None,
        "resources_count": 0,
        "prompts_count": 0,
        "protocol_version": "unknown",
        "has_resources": False,
        "has_prompts": False,
        "has_streaming": False,
        "has_notifications": False,
    }

    try:
        # Test ping latency using the actual ping mechanism
        from chuk_mcp.protocol.messages import send_ping

        streams = list(tm.get_streams())
        if server_index < len(streams):
            read_stream, write_stream = streams[server_index]
            start_time = time.perf_counter()
            try:
                success = await asyncio.wait_for(
                    send_ping(read_stream, write_stream), timeout=5.0
                )
                if success:
                    diagnostics["ping_time"] = (
                        time.perf_counter() - start_time
                    ) * 1000  # Convert to ms
            except:
                pass

        # Test tool list speed
        start_time = time.time()
        if hasattr(tm, "list_tools"):
            tools = await tm.list_tools()
            diagnostics["tool_list_time"] = (time.time() - start_time) * 1000

        # Check for resources
        if hasattr(tm, "list_resources"):
            try:
                resources = await tm.list_resources()
                server_resources = [
                    r for r in resources if r.get("server") == server_name
                ]
                diagnostics["resources_count"] = len(server_resources)
                diagnostics["has_resources"] = len(server_resources) > 0
            except:
                pass

        # Check for prompts
        if hasattr(tm, "list_prompts"):
            try:
                prompts = await tm.list_prompts()
                server_prompts = [p for p in prompts if p.get("server") == server_name]
                diagnostics["prompts_count"] = len(server_prompts)
                diagnostics["has_prompts"] = len(server_prompts) > 0
            except:
                pass

        # Try to get protocol version from initialization data
        if hasattr(tm, "stream_manager") and hasattr(tm.stream_manager, "streams"):
            if server_index < len(tm.stream_manager.streams):
                stream = tm.stream_manager.streams[server_index]
                if hasattr(stream, "protocol_version"):
                    diagnostics["protocol_version"] = stream.protocol_version
                elif hasattr(stream, "_protocol_version"):
                    diagnostics["protocol_version"] = stream._protocol_version
                elif hasattr(stream, "client") and hasattr(
                    stream.client, "protocol_version"
                ):
                    diagnostics["protocol_version"] = stream.client.protocol_version

    except Exception:
        # Silently handle errors to not disrupt display
        pass

    return diagnostics


async def get_server_details(
    tm: ToolManager, server_index: int, config_path: str = "server_config.json"
) -> Dict[str, Any]:
    """Get comprehensive details about a specific server."""
    details = {
        "index": server_index,
        "name": f"server-{server_index}",
        "status": "connected",
        "type": "stdio",  # Default type (stdio, sse, http-sse, websocket)
        "tools": [],
        "tool_count": 0,
        "capabilities": {},
        "config": {},  # Will hold actual server config from JSON
        "command": "unknown",
        "args": [],
        "env": {},
        "enabled": True,
        "connected": False,
        "origin": "user",  # or "built-in"
        "current": False,
    }

    # Load server config to get actual command/args/env
    try:
        from pathlib import Path

        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                import json

                config = json.load(f)
                # Get all server configs
                server_configs = config.get("mcpServers", {})
                # We'll match by name later once we know it
                details["_all_configs"] = server_configs
    except Exception:
        pass

    try:
        # Get server info from ToolManager
        if hasattr(tm, "servers") and tm.servers and server_index < len(tm.servers):
            server = tm.servers[server_index]

            # Get basic info
            if hasattr(server, "name"):
                details["name"] = getattr(server, "name", details["name"])

                # Now get the actual config for this server
                if "_all_configs" in details:
                    server_config = details["_all_configs"].get(details["name"], {})
                    if server_config:
                        details["command"] = server_config.get(
                            "command", details["command"]
                        )
                        details["args"] = server_config.get("args", details["args"])
                        details["env"] = server_config.get("env", details["env"])
                        details["config"] = server_config
                    del details["_all_configs"]

            # Get server type (stdio/sse/http)
            if hasattr(server, "transport"):
                details["type"] = getattr(server, "transport", "stdio")
            elif hasattr(server, "connection_type"):
                details["type"] = getattr(server, "connection_type", "stdio")

            # Get configuration
            if hasattr(server, "config"):
                config = getattr(server, "config", {})
                details["config"] = config
                details["command"] = config.get("command", "unknown")
                details["args"] = config.get("args", [])
                details["env"] = config.get("env", {})
            elif hasattr(server, "command"):
                details["command"] = getattr(server, "command", "unknown")
                if hasattr(server, "args"):
                    details["args"] = getattr(server, "args", [])
                if hasattr(server, "env"):
                    details["env"] = getattr(server, "env", {})
    except Exception:
        pass

    # Get tools for this server using proper ToolManager methods
    try:
        if hasattr(tm, "get_adapted_tools_for_llm"):
            # Use validated tools approach (preferred)
            valid_tools_defs, _ = await tm.get_adapted_tools_for_llm("openai")
            details["tools"] = valid_tools_defs
            details["tool_count"] = len(valid_tools_defs)
        elif hasattr(tm, "list_tools"):
            # Fallback to basic list_tools
            all_tools = await tm.list_tools()
            details["tools"] = all_tools
            details["tool_count"] = len(all_tools)
    except Exception:
        pass

    # Get capabilities
    try:
        # Check various ways capabilities might be stored
        if hasattr(tm, "get_server_capabilities"):
            caps = await tm.get_server_capabilities(server_index)
            if caps:
                details["capabilities"] = caps
        elif hasattr(tm, "servers") and tm.servers and server_index < len(tm.servers):
            server = tm.servers[server_index]
            if hasattr(server, "capabilities"):
                details["capabilities"] = getattr(server, "capabilities", {})
    except Exception:
        pass

    # Set default capabilities if none found
    if not details["capabilities"] and details["tool_count"] > 0:
        details["capabilities"]["tools"] = True

    return details


async def display_servers_table(
    servers: List[Dict[str, Any]], show_details: bool = False
) -> None:
    """Display servers in an enhanced table format similar to theme selector."""

    output.rule("MCP Server Manager")

    table = Table(title="Available Servers", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Server", style="green", width=18)
    table.add_column("Enabled", width=10)
    table.add_column("Status", width=12)
    table.add_column("Tools", justify="center", width=7)
    table.add_column("Type", width=12)
    table.add_column("Origin", width=10)
    table.add_column("Active", justify="center", width=10)

    if show_details:
        table.add_column("Capabilities", width=15)
        table.add_column("Command", width=30)

    for i, server in enumerate(servers, 1):
        # Server name
        server_name = server["name"]

        # Enabled/disabled status
        enabled = server.get("enabled", True)
        enabled_display = "[green]✓[/green]" if enabled else "[red]✗[/red]"

        # Connection status
        connected = server.get("connected", False)
        if connected:
            status_display = "[green]● Connected[/green]"
        elif enabled:
            status_display = "[yellow]● Available[/yellow]"
        else:
            status_display = "[dim]● Disabled[/dim]"

        # Server type (built-in or user)
        origin = server.get("origin", "user")
        if origin == "built-in":
            origin_display = "[blue]Built-in[/blue]"
        else:
            origin_display = "[cyan]User[/cyan]"

        # Transport type (stdio, sse, http-sse, websocket)
        transport = server.get("type", "stdio")
        if transport == "sse" or transport == "http-sse":
            conn_type = "SSE 📡"
        elif transport == "websocket":
            conn_type = "WebSocket 🔌"
        elif transport == "http":
            conn_type = "HTTP 🌐"
        else:
            conn_type = "STDIO 📝"

        # Tool count
        tool_count = server.get("tool_count", 0)
        if tool_count > 0:
            tools_display = f"[green]{tool_count}[/green]"
        else:
            tools_display = "[dim]0[/dim]"

        # Active marker
        is_current = server.get("current", False)
        current_display = "✓ Active" if is_current else ""
        current_style = "bold green" if is_current else "dim"

        row = [
            str(i),
            server_name,
            enabled_display,
            status_display,
            tools_display,
            conn_type,
            origin_display,
            f"[{current_style}]{current_display}[/{current_style}]",
        ]

        if show_details:
            # Capabilities
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

            # Command
            command = server.get("command", "unknown")
            if len(command) > 25:
                command = command[:22] + "..."
            row.append(command)

        table.add_row(*row)

    output.print(table)
    output.print()


async def display_server_detail(server: Dict[str, Any]) -> None:
    """Display detailed view of a single server."""

    output.rule(f"Server Details: {server['name']}")

    # Basic info table
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Property", style="cyan", width=15)
    info_table.add_column("Value", style="white")

    # Status
    enabled = server.get("enabled", True)
    connected = server.get("connected", False)
    if not enabled:
        status = "[dim]Disabled[/dim]"
    elif connected:
        status = "[green]Connected[/green]"
    else:
        status = "[yellow]Available[/yellow]"

    info_table.add_row("Status", status)
    info_table.add_row("Enabled", "Yes" if enabled else "No")
    info_table.add_row(
        "Origin", "Built-in" if server.get("origin") == "built-in" else "User-defined"
    )
    info_table.add_row("Transport", server.get("type", "stdio").upper())
    info_table.add_row("Tools", str(server.get("tool_count", 0)))

    command = server.get("command", "unknown")
    if command != "unknown":
        info_table.add_row("Command", command)

    output.print(Panel(info_table, title="📊 Server Information", border_style="blue"))

    # Show capabilities if available
    caps = server.get("capabilities", {})
    if caps:
        cap_list = []
        if caps.get("tools"):
            cap_list.append("🔧 Tools")
        if caps.get("resources"):
            cap_list.append("📁 Resources")
        if caps.get("prompts"):
            cap_list.append("💬 Prompts")
        if caps.get("logging"):
            cap_list.append("📋 Logging")
        if cap_list:
            output.print()
            output.print(f"[bold]Capabilities:[/bold] {' '.join(cap_list)}")


async def toggle_server_status(server_name: str, enable: bool) -> bool:
    """Enable or disable a server in user preferences."""
    try:
        from mcp_cli.utils.preferences import get_preference_manager

        pref_manager = get_preference_manager()

        if enable:
            pref_manager.enable_server(server_name)
            output.success(f"Server '{server_name}' has been enabled")
            output.info("Server is now enabled and will be available for new sessions")
            output.hint(f"To use immediately: mcp-cli --server {server_name}")
        else:
            pref_manager.disable_server(server_name)
            output.success(f"Server '{server_name}' has been disabled")
            output.info("Server has been disabled in preferences")
            output.hint("This won't affect the current session until you restart")

        return True

    except Exception as e:
        output.error(f"Failed to update preferences: {e}")
        return False


async def servers_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Manage MCP servers - list, add/remove, enable/disable, view details, and test connections.

    Usage:
      /servers                    - List all servers with status
      /servers add <name> [options] - Add a runtime server
      /servers remove <name>      - Remove a runtime server
      /servers <name>             - Show detailed server information
      /servers <name> enable      - Enable a disabled server
      /servers <name> disable     - Disable a server
      /servers <name> config      - Show server configuration
      /servers <name> tools       - List available tools
      /servers <name> ping        - Test server connectivity
      /servers <name> test        - Validate all server tools
      /servers help               - Show this help message

    Examples:
      /servers                    # Show all servers
      /servers add my-server stdio -- npx my-mcp-server
      /servers add api-server http http://localhost:3000
      /servers remove my-server   # Remove runtime server
      /servers sqlite             # Show SQLite server details
      /servers sqlite disable     # Disable SQLite server

    Aliases: /srv"""

    tm: Optional[ToolManager] = ctx.get("tool_manager")
    if tm is None:
        output.error("ToolManager not available")
        return True

    # Parse arguments
    args = parts[1:] if len(parts) > 1 else []

    # Check if user is requesting help
    if args and args[0] in ["help", "--help", "-h"]:
        await show_servers_help()
        return True

    # Handle subcommands: /servers <name> [action]
    if args:
        server_name = args[0]

        # Handle special commands
        if server_name == "add":
            # /servers add <name> [options...]
            if len(args) < 2:
                output.error("Server name required")
                output.hint("Usage: /servers add <name> <transport> [options]")
                output.hint(
                    "Example: /servers add my-server stdio -- npx my-mcp-server"
                )
                output.hint("Example: /servers add api http http://localhost:3000")
                return True
            return await add_runtime_server(args[1:], ctx)

        elif server_name == "remove":
            # /servers remove <name>
            if len(args) < 2:
                output.error("Server name required")
                output.hint("Usage: /servers remove <name>")
                return True
            return await remove_runtime_server(args[1], ctx)

        action = args[1] if len(args) > 1 else "details"

        # Special case: if server_name is a flag, show list
        if server_name.startswith("-"):
            # Legacy flag support
            if server_name in ["--list", "-l"]:
                return await show_servers_list(tm, ctx)
            else:
                output.warning(f"Unknown flag: {server_name}")
                output.hint("Use /servers help for usage information")
                return True

        # Execute server-specific action
        return await handle_server_action(
            tm, ctx, server_name, action, args[2:] if len(args) > 2 else []
        )

    # No arguments - show compact server list
    return await show_servers_list(tm, ctx)


async def show_servers_help() -> None:
    """Display help for servers command."""
    output.rule("Servers Command Help")
    output.print()
    output.print("[bold]Basic Usage:[/bold]")
    output.print(
        "  [cyan]/servers[/cyan]                      - List all servers with available commands"
    )
    output.print(
        "  [cyan]/servers <name>[/cyan]               - Show detailed server information"
    )
    output.print(
        "  [cyan]/srv[/cyan]                          - Short alias for /servers"
    )
    output.print()
    output.print("[bold]Runtime Server Management:[/bold]")
    output.print(
        "  [cyan]/servers add <name> stdio -- <cmd>[/cyan]  - Add STDIO server"
    )
    output.print("  [cyan]/servers add <name> http <url>[/cyan]      - Add HTTP server")
    output.print(
        "  [cyan]/servers remove <name>[/cyan]              - Remove runtime server"
    )
    output.print()
    output.print("[bold]Server Control:[/bold]")
    output.print(
        "  [cyan]/servers <name> enable[/cyan]        - Enable a disabled server"
    )
    output.print("  [cyan]/servers <name> disable[/cyan]       - Disable a server")
    output.print(
        "  [cyan]/servers <name> config[/cyan]        - View server configuration"
    )
    output.print()
    output.print("[bold]Server Testing:[/bold]")
    output.print("  [cyan]/servers <name> tools[/cyan]         - List available tools")
    output.print(
        "  [cyan]/servers <name> ping[/cyan]          - Test server connectivity"
    )
    output.print(
        "  [cyan]/servers <name> test[/cyan]          - Validate all server tools"
    )
    output.print()
    output.print("[bold]Connection:[/bold]")
    output.print(
        "  [cyan]/servers <name> connect[/cyan]       - Show connection instructions"
    )
    output.print()
    output.print("[bold]Examples:[/bold]")
    output.print(
        "  [green]/servers[/green]                      → List all servers with commands"
    )
    output.print(
        "  [green]/servers sqlite[/green]               → Show sqlite server details"
    )
    output.print("  [green]/servers sqlite tools[/green]         → List sqlite tools")
    output.print(
        "  [green]/servers sqlite test[/green]          → Test all sqlite tools"
    )
    output.print(
        "  [green]/servers perplexity disable[/green]   → Disable perplexity server"
    )
    output.print()
    output.print(
        "[bold]Note:[/bold] Use specific commands instead of interactive selection for better workflow."
    )


async def show_servers_list(tm: ToolManager, ctx: Dict[str, Any]) -> bool:
    """Show compact list of all servers."""
    servers = await collect_server_info(tm, ctx)

    if not servers:
        output.warning("No servers configured")
        output.hint("Add servers to server_config.json")
        return True

    # Display servers in a nice table format matching theme command
    output.rule("MCP Servers")

    table = Table(title="Available Servers", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Server", style="green", width=16)
    table.add_column("Status", width=12)
    table.add_column("Enabled", justify="center", width=8)
    table.add_column("Tools", justify="center", width=6)
    table.add_column("R", justify="center", width=3, header_style="dim")  # Resources
    table.add_column("P", justify="center", width=3, header_style="dim")  # Prompts
    table.add_column("Type", width=10)
    table.add_column("Description", style="white")

    # Server descriptions/origins
    server_descriptions = {
        "sqlite": "SQL database server",
        "filesystem": "Local file system access",
        "github": "GitHub repository integration",
        "perplexity": "Perplexity AI search",
        "postgres": "PostgreSQL database server",
        "mysql": "MySQL database server",
        "mongodb": "MongoDB database server",
        "redis": "Redis cache server",
        "default": "Default MCP server",
    }

    for idx, server in enumerate(servers, 1):
        # Status with proper priority: connection state over config state
        connected = server.get("connected", False)
        enabled = server.get("enabled", True)

        if connected:
            # If actually connected, show connected
            status = "[green]● Connected[/green]"
        elif enabled:
            # Enabled in config but not connected
            status = "[yellow]● Available[/yellow]"
        else:
            # Disabled in config and not connected
            status = "[red]● Disabled[/red]"

        # Enabled column - simple yes/no
        enabled_display = "[green]Yes[/green]" if enabled else "[red]No[/red]"

        # Tool count with styling
        tool_count = server.get("tool_count", 0)
        if tool_count > 0:
            tools_display = f"[green]{tool_count}[/green]"
        elif tool_count == -1:
            tools_display = "[yellow]?[/yellow]"
        else:
            tools_display = "[dim]0[/dim]"

        # Type with icon
        server_type_raw = server.get("type", "stdio").lower()
        type_icons = {
            "stdio": "📝 STDIO",
            "sse": "📡 SSE",
            "http": "🌐 HTTP",
            "websocket": "🔌 WebSocket",
        }
        server_type = type_icons.get(server_type_raw, f"❓ {server_type_raw.upper()}")

        # Get resources and prompts indicators (simplified)
        features = server.get("features", {})
        resources_display = "✓" if features.get("resources") else "-"
        prompts_display = "✓" if features.get("prompts") else "-"

        # Get description
        desc = server_descriptions.get(server["name"].lower(), "MCP server")

        table.add_row(
            str(idx),
            server["name"],
            status,
            enabled_display,
            tools_display,
            resources_display,
            prompts_display,
            server_type,
            desc,
        )

    output.print(table)
    output.print("[dim]R = Resources, P = Prompts[/dim]")
    output.print()

    # Show slash command guidance directly
    output.rule("💡 Available Commands")
    output.print("Use these slash commands to manage servers:")
    output.print()

    for server in servers:
        server_name = server["name"]
        tool_count = server.get("tool_count", 0)
        enabled = server.get("enabled", True)

        # Basic commands for all servers
        output.print(f"[bold green]{server_name}:[/bold green]")
        output.print(
            f"  • [cyan]/servers {server_name}[/cyan] - View detailed information"
        )
        output.print(
            f"  • [cyan]/servers {server_name} config[/cyan] - Show configuration"
        )

        if tool_count > 0:
            output.print(
                f"  • [cyan]/servers {server_name} tools[/cyan] - List {tool_count} tools"
            )

        if enabled:
            output.print(
                f"  • [cyan]/servers {server_name} ping[/cyan] - Test connectivity"
            )
            if tool_count > 0:
                output.print(
                    f"  • [cyan]/servers {server_name} test[/cyan] - Validate all tools"
                )
            output.print(
                f"  • [cyan]/servers {server_name} disable[/cyan] - Disable server"
            )
        else:
            output.print(
                f"  • [cyan]/servers {server_name} enable[/cyan] - Enable server"
            )

        output.print()  # Space between servers

    # General commands
    output.print("[bold]General:[/bold]")
    output.print("  • [cyan]/servers help[/cyan] - Show all available commands")
    output.print("  • [cyan]/tools[/cyan] - List all tools from all servers")

    return True


async def handle_server_action(
    tm: ToolManager,
    ctx: Dict[str, Any],
    server_name: str,
    action: str,
    extra_args: List[str],
) -> bool:
    """Handle actions for a specific server."""

    # Get server information
    servers = await collect_server_info(tm, ctx)
    server = None

    # Find the requested server
    for srv in servers:
        if srv["name"].lower() == server_name.lower():
            server = srv
            break

    if not server:
        output.error(f"Server '{server_name}' not found")
        available = ", ".join([s["name"] for s in servers])
        output.hint(f"Available servers: {available}")
        return True

    # Route to appropriate action handler
    action = action.lower()

    if action in ["details", "detail", "info"]:
        return await show_server_details(server, tm, ctx)
    elif action == "enable":
        return await enable_server(server, ctx)
    elif action == "disable":
        return await disable_server(server, ctx)
    elif action in ["config", "configuration"]:
        return await show_server_config(server, ctx)
    elif action == "tools":
        return await show_server_tools(server, tm)
    elif action in ["ping", "connection"]:
        return await ping_server_connection(server, tm)
    elif action == "test":
        return await test_all_server_tools(server, tm)
    elif action in ["connect", "connection-info"]:
        return await show_connection_instructions(server)
    else:
        output.error(f"Unknown action: {action}")
        output.hint(
            "Valid actions: details, enable, disable, config, tools, ping, test, connect"
        )
        return True


async def collect_server_info(
    tm: ToolManager, ctx: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Collect information about all servers."""
    servers = []

    # Load ALL servers from config file (including disabled ones)
    config_path = ctx.get("config_path", "server_config.json")
    all_server_names = set()
    connected_server_names = set()

    # Get connected servers from ToolManager
    connected_servers = {}
    try:
        # Get basic server info from ToolManager (only enabled/connected servers)
        if hasattr(tm, "get_server_info"):
            server_info = await tm.get_server_info()
        else:
            server_info = []

        # If no server info from standard method, try to get from servers list
        if not server_info and hasattr(tm, "servers"):
            server_info = []
            for i, srv in enumerate(tm.servers):
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
        # Don't return here - we still want to show disabled servers

    # Determine which servers are built-in vs user-defined
    # Common built-in servers
    built_in_servers = [
        "filesystem",
        "github",
        "gitlab",
        "google-drive",
        "slack",
        "sqlite",
        "postgres",
        "mysql",
        "mongodb",
        "redis",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "perplexity",
        "youtube-transcripts",
        "ios",
    ]

    # Load config to get ALL servers (including disabled ones)
    config = {}
    try:
        from pathlib import Path

        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                import json

                config = json.load(f)

                # Add all servers from config to our list
                if "mcpServers" in config:
                    for server_name in config["mcpServers"].keys():
                        all_server_names.add(server_name)
    except Exception:
        pass

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
            server_details = await get_server_details(tm, srv_index, config_path)

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
                    server_details["command"] = server_config.get("command", "unknown")
                    server_details["args"] = server_config.get("args", [])
                    server_details["env"] = server_config.get("env", {})
                    server_details["config"] = server_config

                    # Determine type from config
                    if "url" in server_config:
                        server_details["type"] = "http"
                    elif "command" in server_config:
                        server_details["type"] = "stdio"

        # Check if server is in server_config.json (defines origin)
        server_config = config.get("mcpServers", {}).get(server_name, {})

        if server_config:
            # Server is defined in config file - this makes it built-in/configured
            server_details["origin"] = "built-in"
        elif server_name.lower() in built_in_servers:
            # Server is in known built-in list but not explicitly configured
            server_details["origin"] = "built-in"
        else:
            # Unknown/dynamic server (runtime detected)
            server_details["origin"] = "user"

        # Check if server is disabled in preferences (not config)
        server_details["enabled"] = not pref_manager.is_server_disabled(server_name)

        # Special handling for active servers that should have tools available
        if (
            server_name.lower() == "sqlite"
            and server_details.get("connected")
            and server_details.get("tool_count", 0) == 0
        ):
            # If this is the sqlite server and we're probably connected to it
            # This is a heuristic - if we have exactly one connected server and no explicit disconnection
            if len(connected_server_names) == 1:
                # Assume sqlite has tools if it's the only server and supposedly connected
                server_details[
                    "tool_count"
                ] = -1  # Special marker for "unknown but should have tools"

        # Mark as current if it's the only active server
        # (In single server mode, the server is always current)
        if len(connected_server_names) == 1 and server_details.get("connected"):
            server_details["current"] = True

        servers.append(server_details)

    # If still no servers but we have a tool manager with tools, create default entry
    if not servers:
        try:
            if hasattr(tm, "list_tools"):
                tools = await tm.list_tools()
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
                            "enabled": True,
                            "connected": True,
                            "origin": "user",
                            "current": True,
                        }
                    )
        except Exception:
            pass

    return servers


async def show_server_details(
    server: Dict[str, Any], tm: ToolManager, ctx: Optional[Dict[str, Any]] = None
) -> bool:
    """Show detailed information about a server with interactive actions."""
    output.rule(f"Server Details: {server['name']}")

    # Create a nice info table
    from rich.table import Table
    from rich.panel import Panel

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Property", style="bold cyan", width=15)
    info_table.add_column("Value", style="white")

    # Status with proper priority: connection state over config state
    enabled = server.get("enabled", True)
    connected = server.get("connected", False)

    if connected:
        # If actually connected, show connected regardless of config
        status = "[green]● Connected[/green]"
        if not enabled:
            status_desc = " - Active connection (disabled in config)"
        else:
            status_desc = " - Currently active and responding"
    elif enabled:
        # Enabled in config but not connected
        status = "[yellow]● Available[/yellow]"
        status_desc = " - Ready to connect but not currently active"
    else:
        # Disabled in config and not connected
        status = "[red]● Disabled[/red]"
        status_desc = " - Server is disabled in configuration"

    info_table.add_row("Status", status + status_desc)

    # Type with icon
    server_type_raw = server.get("type", "stdio").lower()
    type_icons = {
        "stdio": "📝 STDIO",
        "sse": "📡 SSE",
        "http": "🌐 HTTP",
        "websocket": "🔌 WebSocket",
    }
    server_type = type_icons.get(server_type_raw, f"❓ {server_type_raw.upper()}")
    info_table.add_row("Type", server_type)

    # Tool count with better info - try to get fresh tool count
    tool_count = server.get("tool_count", 0)

    # For connected servers, try to get actual tool count using proper methods
    if connected:
        try:
            if hasattr(tm, "get_adapted_tools_for_llm"):
                # Use validated tools approach (preferred)
                valid_tools_defs, _ = await tm.get_adapted_tools_for_llm("openai")
                if valid_tools_defs:
                    tool_count = len(valid_tools_defs)
                    server["tool_count"] = tool_count  # Update for later use
            elif hasattr(tm, "list_tools"):
                # Fallback to basic list_tools
                actual_tools = await tm.list_tools()
                if actual_tools:
                    tool_count = len(actual_tools)
                    server["tool_count"] = tool_count  # Update for later use
        except Exception:
            pass

    if tool_count > 0:
        if connected:
            tools_display = f"[green]{tool_count} tools available[/green]"
        else:
            tools_display = f"[yellow]{tool_count} tools (when connected)[/yellow]"
    elif tool_count == -1:
        tools_display = f"[yellow]Available when connected[/yellow] - use [cyan]/servers {server['name']} tools[/cyan]"
    else:
        if connected:
            # Connected but no tools detected - could be an issue
            tools_display = f"[yellow]Checking...[/yellow] - use [cyan]/servers {server['name']} tools[/cyan]"
        else:
            tools_display = "[dim]Unknown - server not connected[/dim]"
    info_table.add_row("Tools", tools_display)

    # Origin with better detection
    origin = server.get("origin", "user")
    if origin == "built-in":
        origin_display = (
            "[blue]Built-in server[/blue] - Configured in server_config.json"
        )
    else:
        origin_display = "[cyan]Dynamic server[/cyan] - Detected at runtime"
    info_table.add_row("Origin", origin_display)

    # Get diagnostic data if connected
    diagnostics = {}
    if connected:
        server_index = server.get("index", 0)
        diagnostics = await get_server_diagnostics(tm, server_index, server["name"])

    # Show capabilities if available (enhanced with diagnostic data)
    caps = server.get("capabilities", {})
    features = server.get("features", {})
    cap_list = []

    # Use features from diagnostic data if available
    if (
        diagnostics.get("has_resources")
        or features.get("resources")
        or caps.get("resources")
    ):
        cap_list.append("📁 Resources")
    if diagnostics.get("has_prompts") or features.get("prompts") or caps.get("prompts"):
        cap_list.append("💬 Prompts")
    if tool_count > 0 or features.get("tools") or caps.get("tools"):
        cap_list.append("🔧 Tools")
    if caps.get("logging"):
        cap_list.append("📋 Logging")
    if diagnostics.get("has_streaming") or features.get("streaming"):
        cap_list.append("🌊 Streaming")
    if diagnostics.get("has_notifications") or features.get("notifications"):
        cap_list.append("🔔 Notifications")

    if cap_list:
        info_table.add_row("Capabilities", " ".join(cap_list))

    # Add protocol version if known
    if (
        diagnostics.get("protocol_version")
        and diagnostics["protocol_version"] != "unknown"
    ):
        info_table.add_row("Protocol", f"[dim]{diagnostics['protocol_version']}[/dim]")

    # Display in multiple organized panels

    # Main info panel
    main_panel = Panel(
        info_table, title="📊 Server Information", border_style="blue", padding=(1, 2)
    )
    output.print(main_panel)

    # Additional panels for different aspects
    panels = []

    # Connection details panel if connected (enhanced with diagnostics)
    if connected:
        conn_table = Table(show_header=False, box=None, padding=(0, 1))
        conn_table.add_column("", style="bold yellow", width=12)
        conn_table.add_column("", style="white")

        if tool_count > 0:
            conn_table.add_row("🔧 Tools:", f"{tool_count} available")

        if diagnostics.get("resources_count", 0) > 0:
            conn_table.add_row(
                "📁 Resources:", f"{diagnostics['resources_count']} available"
            )

        if diagnostics.get("prompts_count", 0) > 0:
            conn_table.add_row(
                "💬 Prompts:", f"{diagnostics['prompts_count']} available"
            )

        conn_table.add_row("📡 Status:", "Active connection")

        # Add performance metrics if available
        if diagnostics.get("ping_time") is not None:
            ping_ms = diagnostics["ping_time"]
            if ping_ms < 10:
                ping_display = f"[green]{ping_ms:.1f}ms (Fast)[/green]"
            elif ping_ms < 50:
                ping_display = f"[yellow]{ping_ms:.1f}ms (Good)[/yellow]"
            else:
                ping_display = f"[red]{ping_ms:.1f}ms (Slow)[/red]"
            conn_table.add_row("⏱️  Latency:", ping_display)
        else:
            conn_table.add_row("⏱️  Response:", "Real-time")

        panels.append(
            Panel(
                conn_table,
                title="🟢 Connection Status",
                border_style="green",
                padding=(0, 2),
            )
        )

    # Configuration summary if available
    command = server.get("command", "unknown")
    if command != "unknown":
        config_preview = f"[bold]Command:[/bold] {command}"
        args = server.get("args", [])
        if args:
            # Show first 2 args
            args_preview = " ".join(args[:2])
            if len(args) > 2:
                args_preview += f" ... (+{len(args) - 2} more)"
            config_preview += f"\n[bold]Args:[/bold] {args_preview}"

        panels.append(
            Panel(
                config_preview,
                title="⚙️ Quick Config",
                border_style="cyan",
                padding=(0, 2),
            )
        )

    # Display additional panels in columns if we have them
    if panels:
        output.print()
        for panel in panels:
            output.print(panel)

    output.print()

    # Show available slash commands instead of interactive menu
    output.rule("💡 Available Commands")

    server_name = server["name"]

    # Dynamic command list based on server state
    if enabled:
        if not connected:
            output.print(
                f"🔌 [bold cyan]/servers {server_name} connect[/bold cyan] - Instructions to connect"
            )

        output.print(
            f"⚙️  [bold cyan]/servers {server_name} config[/bold cyan] - View server configuration"
        )

        if tool_count > 0:
            output.print(
                f"🔧 [bold cyan]/servers {server_name} tools[/bold cyan] - List {tool_count} available tools"
            )
        elif tool_count == -1:
            output.print(
                f"🔧 [bold cyan]/servers {server_name} tools[/bold cyan] - Check for available tools"
            )
        else:
            output.print(
                f"🔧 [bold cyan]/servers {server_name} tools[/bold cyan] - Check for available tools"
            )

        output.print(
            f"🏓 [bold cyan]/servers {server_name} ping[/bold cyan] - Ping server connection"
        )
        output.print(
            f"🧪 [bold cyan]/servers {server_name} test[/bold cyan] - Test all tools functionality"
        )
        output.print(
            f"❌ [bold cyan]/servers {server_name} disable[/bold cyan] - Disable this server"
        )
    else:
        output.print(
            f"✅ [bold cyan]/servers {server_name} enable[/bold cyan] - Enable this server"
        )
        output.print(
            f"⚙️  [bold cyan]/servers {server_name} config[/bold cyan] - View server configuration"
        )

    output.print()
    output.print("🔙 [bold cyan]/servers[/bold cyan] - Return to servers list")
    output.print("📚 [bold cyan]/servers help[/bold cyan] - View all server commands")

    return True


async def show_connection_instructions(server: Dict[str, Any]) -> bool:
    """Show instructions for connecting to a server."""
    output.rule(f"Connect to {server['name']}")

    output.print()
    output.print(f"To connect to the [green]{server['name']}[/green] server:")
    output.print()
    output.print(
        "[bold yellow]Option 1:[/bold yellow] Restart MCP CLI with this server"
    )
    output.print(f"[dim]$[/dim] [cyan]mcp-cli --server {server['name']}[/cyan]")
    output.print()

    if server.get("origin") == "user":
        output.print(
            "[bold yellow]Option 2:[/bold yellow] Add to your shell alias or script"
        )
        output.print(
            f"[dim]$[/dim] [cyan]alias mcp-{server['name']}='mcp-cli --server {server['name']}'[/cyan]"
        )
        output.print()

    output.print("[bold yellow]Current Status:[/bold yellow]")
    if server.get("enabled", True):
        output.print("✅ Server is enabled and ready to connect")
        if server.get("connected", False):
            output.print("🔌 Server is already connected in this session")
        else:
            output.print("⏳ Server is not connected in this session")
    else:
        output.print("❌ Server is disabled - enable it first:")
        output.print(f"   [cyan]/servers {server['name']} enable[/cyan]")

    return True


async def enable_server(server: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    """Enable a disabled server."""
    if server.get("enabled", True):
        output.info(f"Server '{server['name']}' is already enabled")
        return True

    await toggle_server_status(server["name"], True)
    return True


async def disable_server(server: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    """Disable a server."""
    if not server.get("enabled", True):
        output.info(f"Server '{server['name']}' is already disabled")
        return True

    if confirm(f"Disable server '{server['name']}'?"):
        await toggle_server_status(server["name"], False)
    return True


async def show_server_config(server: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    """Show server configuration with enhanced formatting."""
    config_path = ctx.get("config_path", "server_config.json")

    output.rule(f"Configuration: {server['name']}")

    # Load the actual config
    actual_config = {}
    try:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                full_config = json.load(f)
                actual_config = full_config.get("mcpServers", {}).get(
                    server["name"], {}
                )
    except Exception:
        pass

    if actual_config:
        # Create a more organized config display
        from rich.panel import Panel
        from rich.table import Table

        # Main config table
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column("Property", style="bold cyan", width=15)
        config_table.add_column("Value", style="white")

        # Extract key information
        command = actual_config.get("command", "Not specified")
        config_table.add_row("Command", f"[green]{command}[/green]")

        args = actual_config.get("args", [])
        if args:
            args_text = "\n".join(f"  • {arg}" for arg in args)
            config_table.add_row("Arguments", args_text)

        env = actual_config.get("env", {})
        if env:
            env_text = "\n".join(f"  • {k}={v}" for k, v in env.items())
            config_table.add_row("Environment", env_text)

        disabled = actual_config.get("disabled", False)
        status = "[red]Disabled[/red]" if disabled else "[green]Enabled[/green]"
        config_table.add_row("Status", status)

        # Additional settings
        extra_settings = {
            k: v
            for k, v in actual_config.items()
            if k not in ["command", "args", "env", "disabled"]
        }
        if extra_settings:
            settings_text = "\n".join(
                f"  • {k}: {v}" for k, v in extra_settings.items()
            )
            config_table.add_row("Other Settings", settings_text)

        # Display in a nice panel
        config_panel = Panel(
            config_table,
            title="⚙️ Server Configuration",
            border_style="cyan",
            padding=(1, 2),
        )
        output.print(config_panel)

        # Show raw JSON in a collapsible way
        output.print()
        output.print("[dim]Raw configuration:[/dim]")
        config_json = json.dumps(actual_config, indent=2)
        syntax = Syntax(
            config_json,
            "json",
            theme="monokai",
            line_numbers=False,
            background_color="default",
        )
        json_panel = Panel(syntax, border_style="dim", padding=(0, 1))
        output.print(json_panel)

        # Show file location
        output.print()
        output.print(f"[dim]Configuration file:[/dim] {config_path}")

    else:
        output.warning(f"No configuration found for '{server['name']}'")
        output.print()
        output.hint(f"Add server configuration to {config_path}")
        output.print("Example configuration:")
        example_config = {
            server["name"]: {
                "command": "your-server-command",
                "args": ["--option", "value"],
                "env": {"KEY": "value"},
            }
        }
        config_json = json.dumps({"mcpServers": example_config}, indent=2)
        syntax = Syntax(config_json, "json", theme="monokai", line_numbers=False)
        example_panel = Panel(
            syntax,
            title="💡 Example Configuration",
            border_style="yellow",
            padding=(0, 1),
        )
        output.print(example_panel)

    return True


async def show_server_tools(
    server: Dict[str, Any], tm: Optional[ToolManager] = None
) -> bool:
    """Show server tools with fresh data from ToolManager."""
    from rich.table import Table

    # Try to get fresh tools from ToolManager if available
    tools = []
    if tm:
        try:
            if hasattr(tm, "get_adapted_tools_for_llm"):
                # Use validated tools approach (preferred)
                valid_tools_defs, _ = await tm.get_adapted_tools_for_llm("openai")
                # Convert tool definitions to display format
                for tool_def in valid_tools_defs:
                    func = tool_def.get("function", {})
                    tools.append(
                        {
                            "name": func.get("name", "unknown"),
                            "description": func.get("description", ""),
                        }
                    )
            elif hasattr(tm, "list_tools"):
                # Fallback to basic list_tools
                all_tools = await tm.list_tools()
                if all_tools:
                    tools = all_tools
        except Exception:
            # Fall back to cached tools
            tools = server.get("tools", [])
    else:
        tools = server.get("tools", [])

    if not tools:
        output.warning(f"No tools available for '{server['name']}'")
        output.hint("Server may need to be connected or may not provide tools")
        return True

    output.rule(f"Available Tools: {server['name']}")

    table = Table(
        title=f"{len(tools)} Tools Available",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Tool Name", style="green", width=25)
    table.add_column("Description", style="white")

    for idx, tool in enumerate(tools, 1):
        if isinstance(tool, dict):
            # Handle both tool info format and function definition format
            if "function" in tool:
                # OpenAI function format
                func = tool["function"]
                tool_name = func.get("name", "unknown")
                tool_desc = func.get("description", "")
            else:
                # Direct tool info format
                tool_name = tool.get("name", "unknown")
                tool_desc = tool.get("description", "")

            if not tool_desc:
                tool_desc = "[dim]No description available[/dim]"
            elif len(tool_desc) > 60:
                tool_desc = tool_desc[:57] + "..."
        else:
            tool_name = str(tool)
            tool_desc = "[dim]No description available[/dim]"

        table.add_row(str(idx), tool_name, tool_desc)

    output.print(table)
    output.print()
    output.hint("Use these tools in chat or with /tools call")

    return True


async def ping_server_connection(server: Dict[str, Any], tm: ToolManager) -> bool:
    """Ping server connection to check if it's responding and measure latency."""
    # Use the shared ping implementation from the main ping command
    from mcp_cli.commands.ping import ping_action_async

    server_name = server["name"]
    server_index = server.get("index", 0)

    # Call the shared ping function for just this one server
    # Use the server name as the target filter
    result = await ping_action_async(tm, targets=[server_name])

    # The ping_action_async already displays the results nicely
    return result


async def ping_server_connection(server: Dict[str, Any], tm: ToolManager) -> bool:
    """Ping a specific server to test connectivity."""
    from chuk_mcp.protocol.messages import send_ping

    server_name = server.get("name", "unknown")

    if not server.get("enabled", True):
        output.warning(f"Server '{server_name}' is disabled")
        output.hint(f"Enable it with: /servers {server_name} enable")
        return True

    if not server.get("connected", False):
        output.warning(f"Server '{server_name}' is not connected")
        output.hint(f"Connect with: mcp-cli --server {server_name}")
        return True

    output.info(f"Pinging server '{server_name}'...")

    # Get the stream for this server
    try:
        # Find server index
        server_index = -1
        if hasattr(tm, "get_server_info"):
            server_info = await tm.get_server_info()
            for i, srv in enumerate(server_info):
                srv_name = srv.name if hasattr(srv, "name") else srv.get("name", "")
                if srv_name == server_name:
                    server_index = i
                    break

        if server_index < 0:
            output.error(f"Could not find server index for '{server_name}'")
            return True

        # Get streams
        streams = tm.get_streams()
        if server_index >= len(streams):
            output.error(f"No stream available for server '{server_name}'")
            return True

        read_stream, write_stream = streams[server_index]

        # Send ping and measure time
        import time

        start_time = time.perf_counter()

        try:
            success = await asyncio.wait_for(
                send_ping(read_stream, write_stream), timeout=5.0
            )
            ping_time = (time.perf_counter() - start_time) * 1000

            if success:
                if ping_time < 10:
                    output.success(f"✅ Ping successful: {ping_time:.1f}ms (Fast)")
                elif ping_time < 50:
                    output.success(f"✅ Ping successful: {ping_time:.1f}ms (Good)")
                elif ping_time < 100:
                    output.warning(f"⚠️  Ping successful: {ping_time:.1f}ms (Slow)")
                else:
                    output.warning(f"⚠️  Ping successful: {ping_time:.1f}ms (Very Slow)")
            else:
                output.error(f"❌ Ping failed for '{server_name}'")

        except asyncio.TimeoutError:
            output.error(f"⏱️  Ping timeout for '{server_name}' (>5s)")
        except Exception as e:
            output.error(f"❌ Ping error: {e}")

    except Exception as e:
        output.error(f"Failed to ping server: {e}")

    return True


async def test_all_server_tools(server: Dict[str, Any], tm: ToolManager) -> bool:
    """Test all tools for a server by attempting to get their schemas."""
    if not server.get("enabled", True):
        output.warning(f"Server '{server['name']}' is disabled")
        output.hint(f"Enable it with: /servers {server['name']} enable")
        return True

    output.info(f"Testing all tools for '{server['name']}'...")

    # Get fresh tools from ToolManager
    tools = []
    try:
        if hasattr(tm, "get_adapted_tools_for_llm"):
            # Use validated tools approach (preferred)
            valid_tools_defs, _ = await tm.get_adapted_tools_for_llm("openai")
            tools = valid_tools_defs
        elif hasattr(tm, "list_tools"):
            # Fallback to basic list_tools
            all_tools = await tm.list_tools()
            if all_tools:
                tools = all_tools
    except Exception as e:
        output.error(f"Failed to get tools: {e}")
        return True

    if not tools:
        output.warning(f"No tools available to test for '{server['name']}'")
        return True

    # Create a test results table
    from rich.table import Table

    output.rule(f"Testing {len(tools)} Tools")

    table = Table(title="Tool Test Results", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Tool Name", style="green", width=25)
    table.add_column("Schema", justify="center", width=8)
    table.add_column("Status", width=12)
    table.add_column("Notes", style="white")

    passed_tests = 0

    for idx, tool in enumerate(tools, 1):
        if isinstance(tool, dict):
            # Handle both tool info format and function definition format
            if "function" in tool:
                # OpenAI function format
                func = tool["function"]
                tool_name = func.get("name", "unknown")
                has_parameters = "parameters" in func and func["parameters"]
                has_description = bool(func.get("description", ""))
            else:
                # Direct tool info format
                tool_name = tool.get("name", "unknown")
                has_parameters = "parameters" in tool or "input_schema" in tool
                has_description = bool(tool.get("description", ""))
        else:
            tool_name = str(tool)
            has_parameters = False
            has_description = False

        # Test results
        schema_status = "✅ Yes" if has_parameters else "⚠️  None"

        if has_description and has_parameters:
            status = "[green]✅ Pass[/green]"
            notes = "Well-defined tool"
            passed_tests += 1
        elif has_description:
            status = "[yellow]⚠️  Partial[/yellow]"
            notes = "Missing parameters schema"
        elif has_parameters:
            status = "[yellow]⚠️  Partial[/yellow]"
            notes = "Missing description"
        else:
            status = "[red]❌ Fail[/red]"
            notes = "Missing description and schema"

        table.add_row(str(idx), tool_name, schema_status, status, notes)

    output.print(table)
    output.print()

    # Summary
    if passed_tests == len(tools):
        output.success(f"🎉 All {len(tools)} tools passed validation!")
    elif passed_tests > 0:
        output.warning(f"⚠️  {passed_tests}/{len(tools)} tools passed validation")
        output.hint("Some tools may need better documentation or schema definitions")
    else:
        output.error(f"❌ {len(tools)} tools failed validation")
        output.hint("Tools may need proper descriptions and parameter schemas")

    return True


async def add_runtime_server(args: List[str], ctx: Dict[str, Any]) -> bool:
    """Add a runtime server to preferences.

    Args format:
      <name> stdio [--env KEY=VALUE] -- <command> [args...]
      <name> http <url>
    """
    if len(args) < 2:
        output.error("Invalid syntax")
        output.hint("Usage: /servers add <name> <transport> [options]")
        return True

    server_name = args[0]
    transport = args[1].lower()

    # Get preference manager
    pref_manager = get_preference_manager()

    # Check if server already exists
    if pref_manager.get_runtime_server(server_name):
        output.error(f"Server '{server_name}' already exists")
        output.hint(f"Use '/servers remove {server_name}' first to replace it")
        return True

    # Parse based on transport type
    config = {"transport": transport}

    if transport == "stdio":
        # Parse stdio server: name stdio [--env KEY=VAL] -- command args...
        env_vars = {}
        command_parts = []
        parsing_command = False

        for i in range(2, len(args)):
            if args[i] == "--":
                parsing_command = True
                continue

            if not parsing_command:
                # Parse environment variables
                if args[i] == "--env" and i + 1 < len(args):
                    # Next arg should be KEY=VALUE
                    env_part = args[i + 1]
                    if "=" in env_part:
                        key, value = env_part.split("=", 1)
                        env_vars[key] = value
                        i += 1  # Skip next arg as we consumed it
            else:
                # Everything after -- is the command
                command_parts.append(args[i])

        if not command_parts:
            output.error("Command required for stdio transport")
            output.hint("Usage: /servers add <name> stdio -- <command> [args...]")
            return True

        config["command"] = command_parts[0]
        config["args"] = command_parts[1:] if len(command_parts) > 1 else []
        if env_vars:
            config["env"] = env_vars

    elif transport == "http" or transport == "sse":
        # Parse HTTP/SSE server: name http <url>
        if len(args) < 3:
            output.error(f"URL required for {transport} transport")
            output.hint(f"Usage: /servers add <name> {transport} <url>")
            return True

        config["url"] = args[2]

    else:
        output.error(f"Unsupported transport: {transport}")
        output.hint("Supported transports: stdio, http, sse")
        return True

    # Add the server to preferences
    pref_manager.add_runtime_server(server_name, config)

    output.success(f"Runtime server '{server_name}' added successfully")
    output.info(f"Transport: {transport}")

    if transport == "stdio":
        output.info(f"Command: {config['command']} {' '.join(config.get('args', []))}")
        if config.get("env"):
            output.info(f"Environment: {config['env']}")
    elif transport in ["http", "sse"]:
        output.info(f"URL: {config['url']}")

    output.hint("Server will be available in new sessions")
    output.hint(f"To use now: Exit and restart with --server {server_name}")

    return True


async def remove_runtime_server(server_name: str, ctx: Dict[str, Any]) -> bool:
    """Remove a runtime server from preferences."""

    # Get preference manager
    pref_manager = get_preference_manager()

    # Check if it's a runtime server
    if not pref_manager.is_runtime_server(server_name):
        # Check if it exists in config file
        config_path = ctx.get("config_path", "server_config.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                if server_name in config.get("mcpServers", {}):
                    output.error(
                        f"'{server_name}' is a configured server, not a runtime server"
                    )
                    output.hint("Runtime servers are those added with '/servers add'")
                    output.hint(
                        f"To disable this server, use: /servers {server_name} disable"
                    )
                    return True
        except:
            pass

        output.error(f"Runtime server '{server_name}' not found")

        # Show available runtime servers
        runtime_servers = pref_manager.get_runtime_servers()
        if runtime_servers:
            output.hint(
                f"Available runtime servers: {', '.join(runtime_servers.keys())}"
            )
        else:
            output.hint("No runtime servers configured")
        return True

    # Remove the server
    if pref_manager.remove_runtime_server(server_name):
        output.success(f"Runtime server '{server_name}' removed")
        output.info("Server will no longer be available in new sessions")
    else:
        output.error(f"Failed to remove server '{server_name}'")

    return True


# Register main command and alias
register_command("/servers", servers_command)
register_command("/srv", servers_command)
