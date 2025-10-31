# src/mcp_cli/commands/actions/servers.py
"""
Servers action for MCP CLI.

List and manage MCP servers with runtime configuration support.
Integrates with preference manager for user servers (~/.mcp-cli)
and config manager for project servers (server_config.json).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Union

from mcp_cli.utils.async_utils import run_blocking
from chuk_term.ui import output, format_table
from mcp_cli.context import get_context
from mcp_cli.config.config_manager import ConfigManager, ServerConfig
from mcp_cli.utils.preferences import get_preference_manager
from mcp_cli.commands.models import ServerActionParams, ServerInfoResponse


def _get_server_icon(capabilities: Dict[str, Any], tool_count: int) -> str:
    """Determine server icon based on MCP capabilities."""
    if capabilities.get("resources") and capabilities.get("prompts"):
        return "🎯"  # Full-featured server
    elif capabilities.get("resources"):
        return "📁"  # Resource-capable server
    elif capabilities.get("prompts"):
        return "💬"  # Prompt-capable server
    elif tool_count > 15:
        return "🔧"  # Tool-heavy server
    elif tool_count > 0:
        return "⚙️"  # Basic tool server
    else:
        return "📦"  # Minimal server


def _format_performance(ping_ms: float | None) -> tuple[str, str]:
    """Format performance metrics with color coding."""
    if ping_ms is None:
        return "❓", "Unknown"

    if ping_ms < 10:
        return "🚀", f"{ping_ms:.1f}ms"
    elif ping_ms < 50:
        return "✅", f"{ping_ms:.1f}ms"
    elif ping_ms < 100:
        return "⚠️", f"{ping_ms:.1f}ms"
    else:
        return "🔴", f"{ping_ms:.1f}ms"


def _format_capabilities(capabilities: Dict[str, Any]) -> str:
    """Format server capabilities as readable string."""
    caps = []

    # Check standard MCP capabilities
    if capabilities.get("tools"):
        caps.append("Tools")
    if capabilities.get("prompts"):
        caps.append("Prompts")
    if capabilities.get("resources"):
        caps.append("Resources")

    # Check experimental capabilities
    experimental = capabilities.get("experimental", {})
    if experimental.get("events"):
        caps.append("Events*")
    if experimental.get("streaming"):
        caps.append("Streaming*")

    return ", ".join(caps) if caps else "None"


def _get_server_status(
    server_config: Union[ServerConfig, Dict[str, Any]], connected: bool = False
) -> tuple[str, str, str]:
    """
    Get server status.
    Returns (status_icon, status_text, status_reason)
    """
    # Handle both ServerConfig objects and dicts
    if isinstance(server_config, ServerConfig):
        disabled = server_config.disabled
        has_command = server_config.command is not None
        has_url = server_config.url is not None
        command = server_config.command
        url = server_config.url
        transport = "http" if has_url else "stdio"
    else:
        disabled = server_config.get("disabled", False)
        has_command = server_config.get("command") is not None
        has_url = server_config.get("url") is not None
        command = server_config.get("command")
        url = server_config.get("url")
        transport = server_config.get("transport", "http")

    if disabled:
        return "⏸️", "Disabled", "Server is disabled"

    if connected:
        return "✅", "Connected", "Server is active"

    # Check if server config is valid
    if not has_command and not has_url:
        return "❌", "Not Configured", "No command or URL specified"

    if has_url:
        # HTTP/SSE server
        return "🌐", transport.upper(), f"URL: {url}"
    elif has_command:
        # STDIO server
        return "📡", "STDIO", f"Command: {command}"

    return "❓", "Unknown", "Unknown server type"


async def _list_servers(show_all: bool = False) -> None:
    """
    List all servers from both config file and runtime preferences.

    Args:
        show_all: Show all servers including disabled ones
    """
    context = get_context()
    tm = context.tool_manager
    pref_manager = get_preference_manager()
    config_manager = ConfigManager()

    # Get project servers from config file
    try:
        config = config_manager.get_config()
    except RuntimeError:
        config = config_manager.initialize()

    # Get runtime servers from preferences
    runtime_servers = pref_manager.get_runtime_servers()

    # Get connected servers from tool manager
    connected_servers = []
    if tm:
        try:
            connected_servers = (
                await tm.get_server_info() if hasattr(tm, "get_server_info") else []
            )
        except Exception:
            pass

    # Track which servers are connected
    connected_names = {s.name.lower() for s in connected_servers}

    # Build combined server list
    table_data = []
    columns = ["Server", "Status", "Transport", "Tools", "Source"]

    # Add project servers from config file
    for name, server_config in config.servers.items():
        if not show_all and server_config.disabled:
            continue

        is_connected = name.lower() in connected_names

        # Get connection info if available
        tool_count = 0
        if is_connected:
            for cs in connected_servers:
                if cs.name.lower() == name.lower():
                    tool_count = cs.tool_count
                    break

        # Convert ServerConfig to dict for status check
        server_dict = {
            "command": server_config.command,
            "url": server_config.url,
            "disabled": server_config.disabled,
            "transport": server_config.transport,
        }

        status_icon, status_text, _ = _get_server_status(server_dict, is_connected)
        display_name = f"→ {name}" if is_connected else f"  {name}"

        table_data.append(
            {
                "Server": display_name,
                "Status": f"{status_icon} {status_text}",
                "Transport": server_config.transport.upper(),
                "Tools": str(tool_count) if tool_count > 0 else "-",
                "Source": "Config",
            }
        )

    # Add runtime servers from preferences
    for name, server_dict in runtime_servers.items():
        # Check if disabled via preferences
        if not show_all and pref_manager.is_server_disabled(name):
            continue

        is_connected = name.lower() in connected_names

        # Get connection info if available
        tool_count = 0
        if is_connected:
            for cs in connected_servers:
                if cs.name.lower() == name.lower():
                    tool_count = cs.tool_count
                    break

        # Add disabled flag if needed
        server_dict["disabled"] = pref_manager.is_server_disabled(name)

        status_icon, status_text, _ = _get_server_status(server_dict, is_connected)
        display_name = f"→ {name}" if is_connected else f"  {name}"

        # Determine transport
        if server_dict.get("url"):
            transport = str(server_dict.get("transport", "HTTP"))
        else:
            transport = "STDIO"

        table_data.append(
            {
                "Server": display_name,
                "Status": f"{status_icon} {status_text}",
                "Transport": transport.upper(),
                "Tools": str(tool_count) if tool_count > 0 else "-",
                "Source": "User",
            }
        )

    # Add any connected servers not in either config or runtime
    for server in connected_servers:
        server_lower = server.name.lower()
        if server_lower not in {
            n.lower() for n in config.servers.keys()
        } and server_lower not in {n.lower() for n in runtime_servers.keys()}:
            table_data.append(
                {
                    "Server": f"→ {server.name}",
                    "Status": "✅ Connected",
                    "Transport": server.transport.upper(),
                    "Tools": str(server.tool_count),
                    "Source": "Active",
                }
            )

    if not table_data:
        output.info("No servers configured.")
        output.tip("Add a server with: /server add <name> stdio <command> [args...]")
        output.tip("Or: /server add <name> http <url>")
        return

    # Display table
    output.rule("[bold]🔌 MCP Servers[/bold]", style="primary")

    table = format_table(
        table_data,
        title=None,
        columns=columns,
    )
    output.print_table(table)
    output.print()

    # Show management tips
    output.tip("💡 Server Management:")
    output.info("  • Add: /server add <name> stdio <command> [args...]")
    output.info("  • Add: /server add <name> --transport http <url>")
    output.info("  • Remove: /server remove <name>")
    output.info("  • Enable/Disable: /server enable|disable <name>")

    # Show counts
    config_count = len(config.servers)
    runtime_count = len(runtime_servers)
    if config_count or runtime_count:
        output.info(
            f"\nServers: {config_count} from config, {runtime_count} user-added"
        )


async def _add_server(
    name: str,
    transport: str,
    config_args: List[str],
    env_vars: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """
    Add a new MCP server to user preferences (~/.mcp-cli).

    Args:
        name: Server name
        transport: Transport type (stdio, http, or sse)
        config_args: Configuration arguments (command+args for stdio, url for http/sse)
        env_vars: Optional environment variables
        headers: Optional headers for HTTP/SSE servers
    """
    pref_manager = get_preference_manager()

    # Check if server already exists
    if pref_manager.get_runtime_server(name):
        output.error(
            f"Server '{name}' already exists. Use '/server update' to modify it."
        )
        return

    # Check in config file too
    config_manager = ConfigManager()
    try:
        config = config_manager.get_config()
        if name in config.servers:
            output.error(
                f"Server '{name}' exists in project config. Remove it first or use a different name."
            )
            return
    except RuntimeError:
        pass

    # Build server configuration as a dictionary for preferences
    server_config: Dict[str, Any] = {"transport": transport.lower()}

    if transport.lower() == "stdio":
        if not config_args:
            output.error("STDIO server requires a command")
            return

        server_config["command"] = config_args[0]
        if len(config_args) > 1:
            server_config["args"] = config_args[1:]
        if env_vars:
            server_config["env"] = env_vars

    elif transport.lower() in ["http", "sse"]:
        if not config_args:
            output.error(f"{transport.upper()} server requires a URL")
            return

        server_config["url"] = config_args[0]
        server_config["transport"] = transport.lower()

        # Store environment variables and headers
        if env_vars:
            server_config["env"] = env_vars
        if headers:
            server_config["headers"] = headers
    else:
        output.error(f"Unknown transport type: {transport}")
        output.info("Valid types: stdio, http, sse")
        return

    # Save to preferences
    pref_manager.add_runtime_server(name, server_config)

    output.success(f"✅ Added server '{name}' to user configuration")
    output.info(f"   Transport: {transport.upper()}")

    if transport.lower() == "stdio":
        output.info(f"   Command: {server_config['command']}")
        if server_config.get("args"):
            output.info(f"   Args: {' '.join(server_config['args'])}")
    else:
        output.info(f"   URL: {server_config['url']}")

    if env_vars:
        output.info(f"   Environment: {', '.join(env_vars.keys())}")

    if headers:
        output.info(f"   Headers: {', '.join(headers.keys())}")

    output.tip(
        "Restart the chat session or use '/server reload' to connect to the new server"
    )


async def _remove_server(name: str) -> None:
    """Remove a server from user preferences or project config."""
    pref_manager = get_preference_manager()

    # Try to remove from runtime servers first
    if pref_manager.remove_runtime_server(name):
        output.success(f"✅ Removed server '{name}' from user configuration")
        output.tip("Restart the chat session to apply changes")
        return

    # Try project config
    config_manager = ConfigManager()
    try:
        config = config_manager.get_config()
    except RuntimeError:
        config = config_manager.initialize()

    if name in config.servers:
        output.warning(f"Server '{name}' is in project configuration.")
        output.info("To remove it, edit server_config.json directly")
        return

    output.error(f"Server '{name}' not found")

    # Show available servers
    runtime_servers = pref_manager.get_runtime_servers()
    all_servers = list(runtime_servers.keys()) + list(config.servers.keys())
    if all_servers:
        output.info(f"Available servers: {', '.join(all_servers)}")


async def _enable_disable_server(name: str, enable: bool) -> None:
    """Enable or disable a server in preferences."""
    pref_manager = get_preference_manager()

    # Check if server exists
    runtime_server = pref_manager.get_runtime_server(name)

    # Also check project config
    config_manager = ConfigManager()
    try:
        config = config_manager.get_config()
        project_server = config.get_server(name)
    except RuntimeError:
        config = config_manager.initialize()
        project_server = None

    if not runtime_server and not project_server:
        output.error(f"Server '{name}' not found")
        return

    # Update preference
    if enable:
        pref_manager.enable_server(name)
        output.success(f"✅ Server '{name}' enabled")
    else:
        pref_manager.disable_server(name)
        output.success(f"✅ Server '{name}' disabled")

    output.tip("Restart the chat session to apply changes")


async def _show_server_details(name: str) -> None:
    """Show detailed information about a server."""
    pref_manager = get_preference_manager()
    config_manager = ConfigManager()

    # Check runtime servers
    server_config = pref_manager.get_runtime_server(name)
    source = "User"

    # Check project config if not found
    if not server_config:
        try:
            config = config_manager.get_config()
            project_server = config.get_server(name)
            if project_server:
                server_config = {
                    "command": project_server.command,
                    "args": project_server.args,
                    "env": project_server.env,
                    "url": project_server.url,
                    "transport": project_server.transport,
                    "disabled": project_server.disabled,
                }
                source = "Config"
        except RuntimeError:
            pass

    if not server_config:
        # Try to show details for a connected server
        context = get_context()
        tm = context.tool_manager
        if tm:
            try:
                servers = (
                    await tm.get_server_info() if hasattr(tm, "get_server_info") else []
                )
                for server in servers:
                    if server.name.lower() == name.lower():
                        await _show_connected_server_details(server)
                        return
            except Exception:
                pass

        output.error(f"Server '{name}' not found")
        return

    # Display detailed info
    is_disabled = pref_manager.is_server_disabled(name) or server_config.get(
        "disabled", False
    )
    icon = "⏸️" if is_disabled else "✅"

    output.rule(f"[bold]{icon} Server: {name}[/bold]", style="primary")
    output.print()

    output.print(f"  [bold]Source:[/bold]      {source}")
    output.print(
        f"  [bold]Status:[/bold]      {'Disabled' if is_disabled else 'Enabled'}"
    )
    output.print(
        f"  [bold]Transport:[/bold]   {server_config.get('transport', 'stdio').upper()}"
    )

    if server_config.get("command"):
        output.print(f"  [bold]Command:[/bold]     {server_config['command']}")
        if server_config.get("args"):
            output.print(
                f"  [bold]Arguments:[/bold]   {' '.join(server_config['args'])}"
            )

    if server_config.get("url"):
        output.print(f"  [bold]URL:[/bold]         {server_config['url']}")

    if server_config.get("env"):
        output.print("  [bold]Environment:[/bold]")
        for key, value in server_config["env"].items():
            # Mask sensitive values
            display_value = (
                "***" if "key" in key.lower() or "token" in key.lower() else value
            )
            output.print(f"    • {key}={display_value}")

    if server_config.get("headers"):
        output.print("  [bold]Headers:[/bold]")
        for key, value in server_config["headers"].items():
            # Mask sensitive headers
            display_value = (
                "***"
                if "auth" in key.lower() or "token" in key.lower()
                else value[:20] + "..."
            )
            output.print(f"    • {key}: {display_value}")

    output.print()

    # Show management options
    if is_disabled:
        output.tip(f"Enable with: /server enable {name}")
    else:
        output.tip(f"Disable with: /server disable {name}")

    if source == "User":
        output.tip(f"Remove with: /server remove {name}")


async def _show_connected_server_details(server) -> None:
    """Show details for a connected server."""
    icon = _get_server_icon(server.capabilities, server.tool_count)

    output.rule(f"[bold]{icon} Server: {server.name}[/bold]", style="primary")
    output.print()

    output.print(f"  [bold]Transport:[/bold]     {server.transport}")
    output.print("  [bold]Status:[/bold]        ✅ Connected")
    output.print(f"  [bold]Tools:[/bold]         {server.tool_count} available")
    output.print(
        f"  [bold]Capabilities:[/bold]  {_format_capabilities(server.capabilities)}"
    )

    output.print()
    output.tip("💡 Use: /servers to list all servers  |  /tools to see available tools")


async def servers_action_async(params: ServerActionParams) -> List[ServerInfoResponse]:
    """
    MCP server management action.

    Args:
        params: Server action parameters

    Example:
        >>> params = ServerActionParams(detailed=True, ping_servers=True)
        >>> await servers_action_async(params)

    Supports:
    - /servers or /server list - List all servers
    - /server add <name> stdio <command> [args...] - Add STDIO server
    - /server add <name> --transport http <url> - Add HTTP/SSE server
    - /server remove <name> - Remove server
    - /server enable <name> - Enable server
    - /server disable <name> - Disable server
    - /server <name> - Show server details
    """

    # Handle command-style invocation with args
    if params.args:
        if not params.args:
            await _list_servers()
            return []

        sub, *rest = params.args
        sub = sub.lower()

        # List servers
        if sub == "list":
            show_all = bool(rest and rest[0].lower() == "all")
            await _list_servers(show_all)
            return []

        # Add server with support for --transport, --env, --header
        if sub == "add" and len(rest) >= 2:
            name = rest[0]

            # Parse options
            transport = "stdio"  # default
            config_args = []
            env_vars = {}
            headers = {}

            i = 1
            while i < len(rest):
                arg = rest[i]

                if arg == "--transport" and i + 1 < len(rest):
                    transport = rest[i + 1]
                    i += 2
                elif arg == "--env" and i + 1 < len(rest):
                    env_str = rest[i + 1]
                    if "=" in env_str:
                        key, value = env_str.split("=", 1)
                        env_vars[key] = value
                    i += 2
                elif arg == "--header" and i + 1 < len(rest):
                    header_str = rest[i + 1]
                    if ":" in header_str:
                        key, value = header_str.split(":", 1)
                        headers[key.strip()] = value.strip()
                    elif "=" in header_str:
                        key, value = header_str.split("=", 1)
                        headers[key.strip()] = value.strip()
                    i += 2
                elif arg == "--":
                    # Everything after -- is the command/URL and args
                    config_args = rest[i + 1 :]
                    break
                else:
                    # First non-option arg could be transport
                    if i == 1 and arg in ["stdio", "http", "sse"]:
                        transport = arg
                        i += 1
                    else:
                        config_args = rest[i:]
                        break

            await _add_server(name, transport, config_args, env_vars, headers)
            return []

        # Remove server
        if sub == "remove" and rest:
            await _remove_server(rest[0])
            return []

        # Enable server
        if sub == "enable" and rest:
            await _enable_disable_server(rest[0], True)
            return []

        # Disable server
        if sub == "disable" and rest:
            await _enable_disable_server(rest[0], False)
            return []

        # Show server details
        await _show_server_details(sub)
        return []

    # Original functionality for backwards compatibility
    context = get_context()
    tm = context.tool_manager

    if not tm:
        output.error("No tool manager available")
        return []

    # Get server information
    try:
        servers = await tm.get_server_info() if hasattr(tm, "get_server_info") else []
    except Exception as e:
        output.error(f"Failed to get server info: {e}")
        return []

    if not servers:
        output.info("No servers connected.")
        output.tip("Add a server with: /server add <name> stdio <command> [args...]")
        return []

    # Process server data
    server_data: List[ServerInfoResponse] = []
    for idx, server in enumerate(servers):
        # ServerInfo is a dataclass with these attributes
        name = server.name
        transport = server.transport
        capabilities = server.capabilities
        tool_count = server.tool_count
        status = server.display_status

        # Ping if requested
        ping_ms = None
        if params.ping_servers:
            try:
                start = time.perf_counter()
                if hasattr(tm, "ping_server"):
                    await tm.ping_server(idx)
                ping_ms = (time.perf_counter() - start) * 1000
            except Exception:
                ping_ms = None

        # Build clean server info response model
        info = ServerInfoResponse(
            name=name,
            transport=transport,
            capabilities=capabilities,
            tool_count=tool_count,
            status=status,
            ping_ms=ping_ms,
        )

        server_data.append(info)

    # Output based on format
    if params.output_format == "json":
        # Convert Pydantic models to dicts for JSON serialization
        output.print(json.dumps([s.model_dump() for s in server_data], indent=2))
    else:
        # Build table
        columns = ["Icon", "Server", "Transport", "Tools", "Capabilities"]
        if params.ping_servers:
            columns.append("Ping")

        table_data: List[Dict[str, Any]] = []
        for server_info in server_data:
            icon = _get_server_icon(server_info.capabilities, server_info.tool_count)
            row: Dict[str, Any] = {
                "Icon": icon,
                "Server": server_info.name,
                "Transport": server_info.transport,
                "Tools": str(server_info.tool_count),
                "Capabilities": _format_capabilities(server_info.capabilities),
            }

            if params.ping_servers:
                perf_icon, perf_text = _format_performance(server_info.ping_ms)
                row["Ping"] = f"{perf_icon} {perf_text}"

            table_data.append(row)

        # Display table with themed styling
        output.rule("[bold]Connected MCP Servers[/bold]", style="primary")

        table = format_table(
            table_data,
            title=None,
            columns=columns,
        )
        output.print_table(table)
        output.print()

        output.tip(
            "💡 Use: /server <name> for details  |  /server add <name> stdio <command>"
        )

    return server_data


def servers_action(**kwargs) -> List[ServerInfoResponse]:
    """
    Sync wrapper for servers_action_async.

    Returns:
        List of server information dictionaries.
    """
    params = ServerActionParams(**kwargs)
    return run_blocking(servers_action_async(params))


__all__ = [
    "servers_action_async",
    "servers_action",
]
