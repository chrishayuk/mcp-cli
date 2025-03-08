# src/cli/chat/commands/servers.py
"""
Commands for working with MCP servers.
"""
from typing import List, Dict, Any
from rich import print
from rich.table import Table
from rich.console import Console

# imports
from cli.chat.commands import register_command

async def cmd_servers(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    List connected MCP servers and their status.
    
    Usage: /servers
    """
    server_info = context['server_info']
    
    servers_table = Table(title="Connected MCP Servers")
    servers_table.add_column("ID", style="cyan")
    servers_table.add_column("Name", style="green")
    servers_table.add_column("Tools", style="cyan")
    servers_table.add_column("Status", style="green")
    
    for server in server_info:
        servers_table.add_row(
            str(server['id']),
            server['name'],
            str(server['tools']),
            server['status']
        )
        
    console = Console()
    console.print(servers_table)
    return True


# Register all commands in this module
register_command("/servers", cmd_servers)