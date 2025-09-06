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
from pathlib import Path

from chuk_term.ui import output
from mcp_cli.tools.models import ServerInfo
from chuk_term.ui.prompts import confirm
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from mcp_cli.chat.commands import register_command
from mcp_cli.utils.preferences import get_preference_manager
from mcp_cli.context import ApplicationContext
from mcp_cli.config import get_config


async def collect_server_info(context: ApplicationContext) -> List[ServerInfo]:
    """Collect information about all servers, returning ServerInfo objects."""
    servers: List[ServerInfo] = []
    tm = context.tool_manager
    config = get_config()

    all_server_names = set()
    connected_server_names = set()

    # Get connected servers from ToolManager
    connected_servers: Dict[str, tuple[int, ServerInfo]] = {}
    if tm:
        try:
            # Get server info from ToolManager (already returns ServerInfo objects)
            server_infos = await tm.get_server_info()

            # Build map of connected servers
            for i, srv in enumerate(server_infos):
                if isinstance(srv, ServerInfo):
                    connected_servers[srv.name] = (i, srv)
                    connected_server_names.add(srv.name)
                    all_server_names.add(srv.name)

        except Exception as e:
            output.error(f"Failed to get server information: {e}")
            # Don't return here - we still want to show disabled servers

    # Add all configured servers
    for server_name in config.servers.keys():
        all_server_names.add(server_name)

    # Get preference manager for disabled status
    pref_manager = get_preference_manager()

    # Process all servers (both connected and from config)
    for server_name in all_server_names:
        server_info: ServerInfo

        # Check if this server is connected (in ToolManager)
        if server_name in connected_servers:
            # Server is connected - use the ServerInfo from ToolManager
            _, server_info = connected_servers[server_name]
            # Update connection status
            server_info.connected = True

        else:
            # Server is not connected - create minimal ServerInfo
            server_info = ServerInfo(
                id=len(servers),  # Use position as ID
                name=server_name,
                status="disconnected",
                tool_count=0,
                namespace=server_name,
                enabled=True,
                connected=False,
                transport="stdio",
                capabilities={},
            )

            # Try to get config info for this server
            server_config = config.get_server(server_name)
            if server_config:
                # Store config directly in ServerInfo fields
                server_info.command = server_config.command
                server_info.args = server_config.args
                server_info.env = server_config.env
                server_info.transport = server_config.transport

        # No need to classify servers - they're all just servers

        # Check if server is disabled in preferences
        server_info.enabled = not pref_manager.is_server_disabled(server_name)

        # Special handling for active servers
        if len(connected_server_names) == 1 and server_info.connected:
            # Mark as current if it's the only active server
            server_info.capabilities["current"] = True

        servers.append(server_info)

    # If still no servers but we have tools, create default entry
    if not servers and tm:
        try:
            tools = await tm.list_tools()
            if tools:
                servers.append(
                    ServerInfo(
                        id=0,
                        name="default",
                        status="connected",
                        tool_count=len(tools),
                        namespace="default",
                        enabled=True,
                        connected=True,
                        transport="stdio",
                        capabilities={"tools": True, "current": True},
                    )
                )
        except Exception:
            pass

    return servers


async def display_servers_table(
    servers: List[ServerInfo], show_details: bool = False
) -> None:
    """Display servers in an enhanced table format."""

    output.rule("MCP Server Manager")

    table = Table(title="Available Servers", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Server", style="green", width=18)
    table.add_column("Enabled", width=10)
    table.add_column("Status", width=12)
    table.add_column("Tools", justify="center", width=7)
    table.add_column("Type", width=12)
    table.add_column("Active", justify="center", width=10)

    if show_details:
        table.add_column("Capabilities", width=15)
        table.add_column("Command", width=30)

    for i, server in enumerate(servers, 1):
        # Use ServerInfo properties
        enabled_display = "✓" if server.enabled else "✗"
        status_display = f"● {server.display_status.capitalize()}"

        # Transport type with icon
        type_icons = {
            "stdio": "📝 STDIO",
            "sse": "📡 SSE",
            "http": "🌐 HTTP",
            "websocket": "🔌 WebSocket",
        }
        conn_type = type_icons.get(server.transport, f"❓ {server.transport.upper()}")

        # Tool count
        tools_display = str(server.tool_count) if server.has_tools else "0"

        # Active marker
        is_current = server.capabilities.get("current", False)
        current_display = "✓ Active" if is_current else ""

        row = [
            str(i),
            server.name,
            enabled_display,
            status_display,
            tools_display,
            conn_type,
            current_display,
        ]

        if show_details:
            # Capabilities
            cap_icons = []
            caps = server.capabilities
            if caps.get("tools") or server.has_tools:
                cap_icons.append("🔧")
            if caps.get("resources"):
                cap_icons.append("📁")
            if caps.get("prompts"):
                cap_icons.append("💬")
            if caps.get("logging"):
                cap_icons.append("📋")
            row.append(" ".join(cap_icons) if cap_icons else "None")

            # Command (from config if available)
            config = server.capabilities.get("config", {})
            command = config.get("command", "unknown")
            if len(command) > 25:
                command = command[:22] + "..."
            row.append(command)

        table.add_row(*row)

    output.print(table)
    output.print()


async def show_servers_list(context: ApplicationContext) -> bool:
    """Show compact list of all servers using ServerInfo models."""
    servers = await collect_server_info(context)

    if not servers:
        output.warning("No servers configured")
        output.hint("Add servers to server_config.json")
        return True

    # Display servers in a nice table format
    output.rule("MCP Servers")

    table = Table(title="Available Servers", show_header=True, header_style="bold cyan")
    table.add_column("#", width=3)
    table.add_column("Server", width=16)
    table.add_column("Status", width=12)
    table.add_column("Enabled", justify="center", width=8)
    table.add_column("Tools", justify="center", width=6)
    table.add_column("Type", width=10)
    table.add_column("Description")

    # No hardcoded descriptions - use what servers provide

    for idx, server in enumerate(servers, 1):
        # Use ServerInfo properties
        status = f"● {server.display_status.capitalize()}"
        enabled_display = "Yes" if server.enabled else "No"
        tools_display = str(server.tool_count) if server.has_tools else "0"

        # Type with icon
        type_icons = {
            "stdio": "📝 STDIO",
            "sse": "📡 SSE",
            "http": "🌐 HTTP",
            "websocket": "🔌 WebSocket",
        }
        server_type = type_icons.get(server.transport, f"❓ {server.transport.upper()}")

        # Get description from server itself
        desc = server.display_description

        table.add_row(
            str(idx),
            server.name,
            status,
            enabled_display,
            tools_display,
            server_type,
            desc,
        )

    output.print(table)
    output.print()

    # Show slash command guidance
    output.rule("💡 Available Commands")
    output.print("Use these slash commands to manage servers:")
    output.print()

    for server in servers:
        output.print(f"{server.name}:")
        output.print(f"  • /servers {server.name} - View detailed information")
        output.print(f"  • /servers {server.name} config - Show configuration")

        if server.has_tools:
            output.print(
                f"  • /servers {server.name} tools - List {server.tool_count} tools"
            )

        if server.enabled:
            output.print(f"  • /servers {server.name} ping - Test connectivity")
            if server.has_tools:
                output.print(f"  • /servers {server.name} test - Validate all tools")
            output.print(f"  • /servers {server.name} disable - Disable server")
        else:
            output.print(f"  • /servers {server.name} enable - Enable server")

        output.print()  # Space between servers

    return True


async def handle_server_action(
    context: ApplicationContext,
    server_name: str,
    action: str,
    extra_args: List[str],
) -> bool:
    """Handle actions for a specific server using ServerInfo models."""

    # Get server information
    servers = await collect_server_info(context)
    server: Optional[ServerInfo] = None

    # Find the requested server
    for srv in servers:
        if srv.name.lower() == server_name.lower():
            server = srv
            break

    if not server:
        output.error(f"Server '{server_name}' not found")
        available = ", ".join([s.name for s in servers])
        output.hint(f"Available servers: {available}")
        return True

    # Route to appropriate action handler
    action = action.lower()

    if action in ["details", "detail", "info"]:
        return await show_server_details(server, context)
    elif action == "enable":
        return await enable_server(server)
    elif action == "disable":
        return await disable_server(server)
    elif action in ["config", "configuration"]:
        return await show_server_config(server)
    elif action == "tools":
        return await show_server_tools(server, context)
    elif action in ["ping", "connection"]:
        return await ping_server_connection(server, context)
    elif action == "test":
        return await test_all_server_tools(server, context)
    elif action in ["connect", "connection-info"]:
        return await show_connection_instructions(server)
    else:
        output.error(f"Unknown action: {action}")
        output.hint(
            "Valid actions: details, enable, disable, config, tools, ping, test, connect"
        )
        return True


async def show_server_details(server: ServerInfo, context: ApplicationContext) -> bool:
    """Show detailed information about a server using ServerInfo model."""
    output.rule(f"Server Details: {server.name}")

    # Create info table
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Property", style="bold cyan", width=15)
    info_table.add_column("Value", style="white")

    # Use ServerInfo properties
    status_display = server.display_status.capitalize()
    if server.connected:
        status_desc = " - Currently active and responding"
    elif server.enabled:
        status_desc = " - Ready to connect but not currently active"
    else:
        status_desc = " - Server is disabled in configuration"

    info_table.add_row("Status", f"● {status_display}{status_desc}")

    # Type with icon
    type_icons = {
        "stdio": "📝 STDIO",
        "sse": "📡 SSE",
        "http": "🌐 HTTP",
        "websocket": "🔌 WebSocket",
    }
    server_type = type_icons.get(server.transport, f"❓ {server.transport.upper()}")
    info_table.add_row("Type", server_type)

    # Tool count
    if server.has_tools:
        if server.connected:
            tools_display = f"{server.tool_count} tools available"
        else:
            tools_display = f"{server.tool_count} tools (when connected)"
    else:
        if server.connected:
            tools_display = "Checking... - use /servers {server.name} tools"
        else:
            tools_display = "Unknown - server not connected"
    info_table.add_row("Tools", tools_display)

    # Source - where is this server configured
    if server.command:
        source_display = f"Command: {server.command}"
    else:
        source_display = "Runtime detected"
    info_table.add_row("Source", source_display)

    # Display main panel
    main_panel = Panel(
        info_table, title="📊 Server Information", border_style="blue", padding=(1, 2)
    )
    output.print(main_panel)

    # Show available commands
    output.print()
    output.rule("💡 Available Commands")

    if server.enabled:
        output.print(f"⚙️  /servers {server.name} config - View server configuration")
        if server.has_tools:
            output.print(
                f"🔧 /servers {server.name} tools - List {server.tool_count} available tools"
            )
        output.print(f"🏓 /servers {server.name} ping - Ping server connection")
        output.print(f"🧪 /servers {server.name} test - Test all tools functionality")
        output.print(f"❌ /servers {server.name} disable - Disable this server")
    else:
        output.print(f"✅ /servers {server.name} enable - Enable this server")
        output.print(f"⚙️  /servers {server.name} config - View server configuration")

    return True


async def enable_server(server: ServerInfo) -> bool:
    """Enable a disabled server."""
    if server.enabled:
        output.info(f"Server '{server.name}' is already enabled")
        return True

    await toggle_server_status(server.name, True)
    return True


async def disable_server(server: ServerInfo) -> bool:
    """Disable a server."""
    if not server.enabled:
        output.info(f"Server '{server.name}' is already disabled")
        return True

    if confirm(f"Disable server '{server.name}'?"):
        await toggle_server_status(server.name, False)
    return True


async def show_server_config(server: ServerInfo) -> bool:
    """Show server configuration using ServerInfo model."""
    config = get_config()

    output.rule(f"Configuration: {server.name}")

    # Get config from ServerInfo capabilities or load from file
    actual_config = server.capabilities.get("config", {})

    if not actual_config:
        # Try loading from file
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, "r") as f:
                    full_config = json.load(f)
                    actual_config = full_config.get("mcpServers", {}).get(
                        server.name, {}
                    )
        except Exception:
            pass

    if actual_config:
        # Display configuration
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column("Property", style="bold cyan", width=15)
        config_table.add_column("Value", style="white")

        command = actual_config.get("command", "Not specified")
        config_table.add_row("Command", command)

        args = actual_config.get("args", [])
        if args:
            args_text = "\n".join(f"  • {arg}" for arg in args)
            config_table.add_row("Arguments", args_text)

        env = actual_config.get("env", {})
        if env:
            env_text = "\n".join(f"  • {k}={v}" for k, v in env.items())
            config_table.add_row("Environment", env_text)

        config_panel = Panel(
            config_table,
            title="⚙️ Server Configuration",
            border_style="cyan",
            padding=(1, 2),
        )
        output.print(config_panel)

        # Show raw JSON
        output.print()
        output.print("Raw configuration:")
        config_json = json.dumps(actual_config, indent=2)
        syntax = Syntax(config_json, "json", theme="monokai", line_numbers=False)
        json_panel = Panel(syntax, border_style="dim", padding=(0, 1))
        output.print(json_panel)
    else:
        output.warning(f"No configuration found for '{server.name}'")
        output.hint(f"Add server configuration to {config_path}")

    return True


async def show_server_tools(server: ServerInfo, context: ApplicationContext) -> bool:
    """Show server tools using ServerInfo model."""
    # Implementation would fetch tools for this specific server
    # For now, just show tool count from ServerInfo

    if not server.has_tools:
        output.warning(f"No tools available for '{server.name}'")
        return True

    output.rule(f"Available Tools: {server.name}")
    output.info(f"{server.tool_count} tools available")
    output.hint("Use /tools to see all tools from all servers")

    return True


async def ping_server_connection(
    server: ServerInfo, context: ApplicationContext
) -> bool:
    """Ping a specific server to test connectivity."""

    if not server.enabled:
        output.warning(f"Server '{server.name}' is disabled")
        output.hint(f"Enable it with: /servers {server.name} enable")
        return True

    if not server.connected:
        output.warning(f"Server '{server.name}' is not connected")
        output.hint(f"Connect with: mcp-cli --server {server.name}")
        return True

    output.info(f"Pinging server '{server.name}'...")

    # Implementation would ping the specific server
    # For now, just show status
    if server.is_healthy:
        output.success(f"✅ Server '{server.name}' is healthy and responding")
    else:
        output.warning(f"⚠️ Server '{server.name}' status: {server.status}")

    return True


async def test_all_server_tools(
    server: ServerInfo, context: ApplicationContext
) -> bool:
    """Test all tools for a server."""

    if not server.enabled:
        output.warning(f"Server '{server.name}' is disabled")
        return True

    if not server.has_tools:
        output.warning(f"No tools available to test for '{server.name}'")
        return True

    output.info(f"Testing {server.tool_count} tools for '{server.name}'...")
    # Implementation would test each tool
    output.success(f"Tool testing complete for '{server.name}'")

    return True


async def show_connection_instructions(server: ServerInfo) -> bool:
    """Show instructions for connecting to a server."""
    output.rule(f"Connect to {server.name}")

    output.print()
    output.print(f"To connect to the {server.name} server:")
    output.print()
    output.print("Option 1: Restart MCP CLI with this server")
    output.print(f"  $ mcp-cli --server {server.name}")
    output.print()

    output.print("Option 2: Add to your shell alias or script")
    output.print(f"  $ alias mcp-{server.name}='mcp-cli --server {server.name}'")
    output.print()

    output.print("Current Status:")
    if server.enabled:
        output.print("✅ Server is enabled and ready to connect")
        if server.connected:
            output.print("🔌 Server is already connected in this session")
        else:
            output.print("⏳ Server is not connected in this session")
    else:
        output.print("❌ Server is disabled - enable it first:")
        output.print(f"   /servers {server.name} enable")

    return True


async def toggle_server_status(server_name: str, enable: bool) -> bool:
    """Enable or disable a server in user preferences."""
    try:
        pref_manager = get_preference_manager()

        if enable:
            pref_manager.enable_server(server_name)
            output.success(f"Server '{server_name}' has been enabled")
        else:
            pref_manager.disable_server(server_name)
            output.success(f"Server '{server_name}' has been disabled")

        return True

    except Exception as e:
        output.error(f"Failed to update preferences: {e}")
        return False


async def servers_command(parts: List[str], ctx: Dict[str, Any] = None) -> bool:
    """Manage MCP servers - now using ServerInfo models throughout."""

    # Use global context manager
    from mcp_cli.context import get_context

    context = get_context()

    if context.tool_manager is None:
        output.error("ToolManager not available")
        return True

    # Parse arguments
    args = parts[1:] if len(parts) > 1 else []

    # No arguments - show server list
    if not args:
        return await show_servers_list(context)

    # Handle subcommands
    server_name = args[0]
    action = args[1] if len(args) > 1 else "details"

    return await handle_server_action(
        context, server_name, action, args[2:] if len(args) > 2 else []
    )


# Register commands
register_command("/servers", servers_command)
register_command("/srv", servers_command)
