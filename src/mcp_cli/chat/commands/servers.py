from __future__ import annotations

"""
Enhanced /servers command with detailed capability and protocol information.

Usage Examples
--------------
/servers                    - Basic table view with feature icons
/servers --detailed         - Full detailed view with panels for each server
/servers --capabilities     - Include capability details in output
/servers --transport        - Show transport/connection information  
/servers --tree             - Display in tree format
/servers --json             - Raw JSON output for programmatic use
/srv -d                     - Short alias with detailed flag
"""

from typing import Any, Dict, List

from mcp_cli.utils.rich_helpers import get_console
from mcp_cli.commands.servers import servers_action_async
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command


async def servers_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Enhanced server information display with comprehensive details."""
    console = get_console()

    tm: ToolManager | None = ctx.get("tool_manager")
    if tm is None:
        console.print("[red]Error:[/red] ToolManager not available.")
        return True

    # Parse arguments
    args = parts[1:]  # Remove command name
    
    # Parse flags
    detailed = any(arg in ["--detailed", "-d", "--detail"] for arg in args)
    show_capabilities = any(arg in ["--capabilities", "--caps", "-c"] for arg in args)
    show_transport = any(arg in ["--transport", "--trans", "-t"] for arg in args)
    
    # Output format
    output_format = "table"  # default
    if "--json" in args:
        output_format = "json"
    elif "--tree" in args:
        output_format = "tree"
    
    # If detailed is requested, automatically enable capabilities and transport
    if detailed:
        show_capabilities = True
        show_transport = True
    
    await servers_action_async(
        tm,
        detailed=detailed,
        show_capabilities=show_capabilities,
        show_transport=show_transport,
        output_format=output_format
    )
    
    return True


# Register main command and alias
register_command("/servers", servers_command)
register_command("/srv", servers_command)