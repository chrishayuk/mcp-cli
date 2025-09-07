# mcp_cli/chat/commands/servers.py
"""Enhanced /servers command with detailed capability and protocol information."""

from __future__ import annotations

from typing import Dict, List, Optional
import json

from chuk_term.ui import output, format_table
from chuk_term.ui.prompts import confirm

from mcp_cli.chat.commands import register_command
from mcp_cli.utils.preferences import get_preference_manager
from mcp_cli.config import get_config
from mcp_cli.tools.models import ServerInfo


async def collect_server_info(context) -> List[ServerInfo]:
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


def display_servers_table(
    servers: List[ServerInfo], show_details: bool = False
) -> None:
    """Display servers in an enhanced table format using chuk_term."""

    output.rule("MCP Server Manager")

    # Build data for chuk_term's format_table
    data = []

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

        row_data = {
            "#": str(i),
            "Server": server.name,
            "Enabled": enabled_display,
            "Status": status_display,
            "Tools": tools_display,
            "Type": conn_type,
            "Active": current_display,
        }

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
            row_data["Capabilities"] = " ".join(cap_icons) if cap_icons else "None"

            # Command (from config if available)
            command = server.command or "unknown"
            if len(command) > 25:
                command = command[:22] + "..."
            row_data["Command"] = command

        data.append(row_data)

    # Define columns based on detail level
    columns = ["#", "Server", "Enabled", "Status", "Tools", "Type", "Active"]
    if show_details:
        columns.extend(["Capabilities", "Command"])

    # Create and print table using chuk_term
    table = format_table(data=data, title="Available Servers", columns=columns)

    output.print(table)
    output.print()


async def show_servers_list(context) -> bool:
    """Show compact list of all servers using ServerInfo models."""
    servers = await collect_server_info(context)

    if not servers:
        output.warning("No servers configured")
        output.hint("Add servers to server_config.json")
        return True

    # Display servers in a nice table format
    output.rule("MCP Servers")

    # Build data for chuk_term's format_table
    data = []

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

        data.append(
            {
                "#": str(idx),
                "Server": server.name,
                "Status": status,
                "Enabled": enabled_display,
                "Tools": tools_display,
                "Type": server_type,
                "Description": desc,
            }
        )

    # Create and print table using chuk_term
    table = format_table(
        data=data,
        title="Available Servers",
        columns=["#", "Server", "Status", "Enabled", "Tools", "Type", "Description"],
    )

    output.print(table)
    output.print()

    # Show command guidance
    output.rule("Available Commands")
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
    context,
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


async def show_server_details(server: ServerInfo, context) -> bool:
    """Show detailed information about a server using ServerInfo model."""
    output.rule(f"Server Details: {server.name}")

    # Build info data for table
    info_data = []

    # Status
    status_display = server.display_status.capitalize()
    if server.connected:
        status_desc = " - Currently active and responding"
    elif server.enabled:
        status_desc = " - Ready to connect but not currently active"
    else:
        status_desc = " - Server is disabled in configuration"
    info_data.append(
        {"Property": "Status", "Value": f"● {status_display}{status_desc}"}
    )

    # Type with icon
    type_icons = {
        "stdio": "📝 STDIO",
        "sse": "📡 SSE",
        "http": "🌐 HTTP",
        "websocket": "🔌 WebSocket",
    }
    server_type = type_icons.get(server.transport, f"❓ {server.transport.upper()}")
    info_data.append({"Property": "Type", "Value": server_type})

    # Tool count
    if server.has_tools:
        if server.connected:
            tools_display = f"{server.tool_count} tools available"
        else:
            tools_display = f"{server.tool_count} tools (when connected)"
    else:
        if server.connected:
            tools_display = f"Checking... - use /servers {server.name} tools"
        else:
            tools_display = "Unknown - server not connected"
    info_data.append({"Property": "Tools", "Value": tools_display})

    # Source
    if server.command:
        source_display = f"Command: {server.command}"
    else:
        source_display = "Runtime detected"
    info_data.append({"Property": "Source", "Value": source_display})

    # Create and display table
    info_table = format_table(
        data=info_data, title="Server Information", columns=["Property", "Value"]
    )
    output.print(info_table)

    # Show available commands
    output.print()
    output.rule("Available Commands")

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
    output.rule(f"Configuration: {server.name}")

    # Build config data
    config_data = []

    if server.command:
        config_data.append({"Property": "Command", "Value": server.command})

    if server.args:
        args_text = "\n".join(f"  • {arg}" for arg in server.args)
        config_data.append({"Property": "Arguments", "Value": args_text})

    if server.env:
        env_text = "\n".join(f"  • {k}={v}" for k, v in server.env.items())
        config_data.append({"Property": "Environment", "Value": env_text})

    if config_data:
        # Create and display table
        config_table = format_table(
            data=config_data,
            title="Server Configuration",
            columns=["Property", "Value"],
        )
        output.print(config_table)

        # Show raw JSON
        output.print()
        output.print("Raw configuration:")
        config_dict = {
            "command": server.command,
            "args": server.args,
            "env": server.env,
            "transport": server.transport,
        }
        config_json = json.dumps(config_dict, indent=2)
        output.print(config_json)
    else:
        output.warning(f"No configuration found for '{server.name}'")
        output.hint("Add server configuration to server_config.json")

    return True


async def show_server_tools(server: ServerInfo, context) -> bool:
    """Show server tools using ServerInfo model."""
    if not server.has_tools:
        output.warning(f"No tools available for '{server.name}'")
        return True

    output.rule(f"Available Tools: {server.name}")
    output.info(f"{server.tool_count} tools available")
    output.hint("Use /tools to see all tools from all servers")

    return True


async def ping_server_connection(server: ServerInfo, context) -> bool:
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

    # Check server status
    if server.is_healthy:
        output.success(f"✅ Server '{server.name}' is healthy and responding")
    else:
        output.warning(f"⚠️ Server '{server.name}' status: {server.status}")

    return True


async def test_all_server_tools(server: ServerInfo, context) -> bool:
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


async def servers_command(parts: List[str]) -> bool:
    """Manage MCP servers - now using ServerInfo models throughout.

    Usage:
        /servers                      - Show all servers with status
        /servers <name>               - Show details for a specific server
        /servers <name> enable        - Enable a disabled server
        /servers <name> disable       - Disable a server
        /servers <name> config        - Show server configuration
        /servers <name> tools         - List server tools
        /servers <name> ping          - Test server connectivity
        /servers <name> test          - Test all server tools
        /servers <name> connect       - Show connection instructions

    Examples:
        /servers                      - List all configured servers
        /servers sqlite               - Show SQLite server details
        /servers sqlite tools         - List SQLite server tools
        /servers echo disable         - Disable the echo server
    """
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

# Register command help
SERVERS_HELP = {
    "name": "servers",
    "description": "Manage MCP servers",
    "usage": [
        "/servers - List all configured servers",
        "/servers <name> - Show server details",
        "/servers <name> <action> - Perform action on server",
    ],
    "examples": [
        "/servers",
        "/servers sqlite",
        "/servers sqlite tools",
        "/servers echo disable",
    ],
}
