"""
Tools command module for listing available tools with their server sources.
"""
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
import json

# Import the registration function from the command handler
from mcp_cli.chat.commands import register_command

async def tools_command(cmd_parts, context):
    """
    Display all available tools with their server information.
    
    Usage:
      /tools         - Show tools with descriptions
      /tools --all   - Show all tool details including parameters
      /tools --raw   - Show raw tool definitions (for debugging)
    
    This command shows all tools available across all connected servers,
    making it clear which server provides each tool.
    """
    console = Console()
    
    # Parse arguments - skip the command name itself
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
    
    # Parse options
    show_all = "--all" in args
    show_raw = "--raw" in args
    
    # Get tools and server mapping
    tools = context.get("tools", [])
    tool_to_server_map = context.get("tool_to_server_map", {})
    
    if not tools:
        console.print("[yellow]No tools available.[/yellow]")
        return True
    
    # Show raw JSON if --raw was given
    if show_raw:
        raw_json = json.dumps(tools, indent=2)
        console.print(Syntax(raw_json, "json", theme="monokai", line_numbers=True))
        return True
    
    # Create and populate a rich table
    table = Table(title=f"{len(tools)} Available Tools")
    table.add_column("Server", style="cyan")
    table.add_column("Tool", style="green")
    table.add_column("Description")
    
    if show_all:
        table.add_column("Parameters", style="yellow")
    
    for tool in tools:
        tool_name = tool.get("name", "unknown")
        server_name = tool_to_server_map.get(tool_name, "Unknown")
        
        # Possibly truncate the description if --all not used
        description = tool.get("description", "No description")
        if not show_all and len(description) > 75:
            description = description[:72] + "..."
        
        if show_all:
            # Handle multiple schema formats (OpenAI / MCP)
            parameters = tool.get("parameters", tool.get("inputSchema", {}))
            
            if "properties" in parameters:
                properties = parameters["properties"]
                required = parameters.get("required", [])
            elif isinstance(parameters, dict):
                properties = parameters.get("properties", {})
                required = parameters.get("required", [])
            else:
                properties = {}
                required = []
            
            # Generate parameter info
            param_strs = []
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                is_required = "*" if param_name in required else ""
                param_strs.append(f"{param_name}{is_required} ({param_type})")
            
            param_text = "\n".join(param_strs) if param_strs else "None"
            table.add_row(server_name, tool_name, description, param_text)
        else:
            table.add_row(server_name, tool_name, description)
    
    console.print(table)
    
    if show_all:
        console.print("[yellow]* Required parameter[/yellow]")
    
    return True

# Register the command with its options
register_command("/tools", tools_command, ["--all", "--raw"])