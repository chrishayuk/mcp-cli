# src/mcp_cli/commands/actions/servers.py
"""
Servers action for MCP CLI.

List and manage connected MCP servers with enhanced capabilities.

Public functions:
* **servers_action_async()** - Async function to list server information.
* **servers_action()** - Sync wrapper for legacy CLI paths.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking
from chuk_term.ui import output, format_table
from mcp_cli.context import get_context


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


async def servers_action_async(
    detailed: bool = False,
    show_capabilities: bool = False,
    show_transport: bool = False,
    output_format: str = "table",
    ping_servers: bool = False,
) -> List[Dict[str, Any]]:
    """
    Display information about connected MCP servers.
    
    Args:
        detailed: Show detailed server information.
        show_capabilities: Include server capabilities.
        show_transport: Include transport details.
        output_format: Output format ('table' or 'json').
        ping_servers: Test server connectivity.
    
    Returns:
        List of server information dictionaries.
    """
    context = get_context()
    tm = context.tool_manager
    
    if not tm:
        output.error("No tool manager available")
        return []
    
    # Get server information
    try:
        servers = await tm.get_server_info() if hasattr(tm, 'get_server_info') else []
    except Exception as e:
        output.error(f"Failed to get server info: {e}")
        return []
    
    if not servers:
        output.info("No servers connected.")
        return []
    
    # Process server data using ServerInfo model
    server_data = []
    for idx, server in enumerate(servers):
        # ServerInfo is a dataclass with these attributes
        name = server.name
        transport = server.transport
        capabilities = server.capabilities
        tool_count = server.tool_count
        status = server.display_status
        
        # Ping if requested
        ping_ms = None
        if ping_servers:
            try:
                start = time.perf_counter()
                if hasattr(tm, 'ping_server'):
                    await tm.ping_server(idx)
                ping_ms = (time.perf_counter() - start) * 1000
            except:
                ping_ms = None
        
        # Build clean server info dict for display
        info = {
            "name": name,
            "transport": transport,
            "capabilities": capabilities,
            "tool_count": tool_count,
            "status": status,
            "ping_ms": ping_ms,
        }
        
        server_data.append(info)
    
    # Output based on format
    if output_format == "json":
        output.print(json.dumps(server_data, indent=2))
    else:
        # Build table
        columns = ["Icon", "Server", "Transport", "Tools", "Capabilities"]
        if ping_servers:
            columns.append("Ping")
        
        table_data = []
        for server in server_data:
            icon = _get_server_icon(server["capabilities"], server["tool_count"])
            row = {
                "Icon": icon,
                "Server": server["name"],
                "Transport": server["transport"],
                "Tools": str(server["tool_count"]),
                "Capabilities": _format_capabilities(server["capabilities"]),
            }
            
            if ping_servers:
                perf_icon, perf_text = _format_performance(server["ping_ms"])
                row["Ping"] = f"{perf_icon} {perf_text}"
            
            table_data.append(row)
        
        # Display table with themed styling
        output.rule("[bold]Connected MCP Servers[/bold]", style="primary")
        
        table = format_table(
            table_data,
            title=None,  # We're using rule for title
            columns=columns
        )
        output.print_table(table)
        
        # Add some visual separation
        output.print()
        
        # Show tip with icon
        output.tip("💡 Use: /server <name> for details  |  /ping to test connectivity  |  /tools to see available tools")
    
    return server_data


def servers_action(**kwargs) -> List[Dict[str, Any]]:
    """
    Sync wrapper for servers_action_async.
    
    Returns:
        List of server information dictionaries.
        
    Raises:
        RuntimeError: If called from inside an active event loop.
    """
    return run_blocking(servers_action_async(**kwargs))


async def server_details_async(server_name: str) -> None:
    """
    Show detailed information about a specific server.
    
    Args:
        server_name: Name of the server to show details for.
    """
    context = get_context()
    tm = context.tool_manager
    
    if not tm:
        output.error("No tool manager available")
        return
    
    # Get server information
    try:
        servers = await tm.get_server_info() if hasattr(tm, 'get_server_info') else []
    except Exception as e:
        output.error(f"Failed to get server info: {e}")
        return
    
    # Find the specific server
    target_server = None
    for server in servers:
        if server.name.lower() == server_name.lower():
            target_server = server
            break
    
    if not target_server:
        output.error(f"Server not found: {server_name}")
        output.hint("Use /servers to list all available servers")
        return
    
    # Display detailed server info with visual appeal
    icon = _get_server_icon(target_server.capabilities, target_server.tool_count)
    
    # Ping the server
    ping_ms = None
    try:
        start = time.perf_counter()
        if hasattr(tm, 'ping_server'):
            server_idx = servers.index(target_server)
            await tm.ping_server(server_idx)
        ping_ms = (time.perf_counter() - start) * 1000
        ping_status = f"✅ Online ({ping_ms:.1f}ms)"
    except:
        ping_status = "❓ Unknown"
    
    # Display with visual formatting
    output.rule(f"[bold]{icon} Server: {target_server.name}[/bold]", style="primary")
    output.print()
    
    output.print(f"  [bold]Transport:[/bold]     {target_server.transport}")
    output.print(f"  [bold]Status:[/bold]        {ping_status}")
    output.print(f"  [bold]Tools:[/bold]         {target_server.tool_count} available")
    output.print(f"  [bold]Capabilities:[/bold]  {_format_capabilities(target_server.capabilities)}")
    
    # Add tools list if available and not too many
    if target_server.tool_count > 0 and target_server.tool_count <= 10:
        try:
            tools = await tm.get_tools_for_server(server_name) if hasattr(tm, 'get_tools_for_server') else []
            if tools:
                output.print()
                output.print("  [bold]Available tools:[/bold]")
                for tool in tools[:10]:
                    tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                    output.print(f"    • {tool_name}")
        except:
            pass
    
    # Show tip with spacing
    output.print()
    output.tip("💡 Use: /servers to list all servers  |  /ping to test connectivity")


__all__ = [
    "servers_action_async",
    "servers_action",
    "server_details_async",
]