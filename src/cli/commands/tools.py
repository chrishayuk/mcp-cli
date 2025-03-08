# src/cli/chat/commands/tools.py
"""
Tools command module for listing available tools with their server sources.
"""
from rich.console import Console
from rich.table import Table

# imports
from cli.chat.commands import register_command


async def tools_command(args, context):
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
    
    # Parse arguments
    show_all = "--all" in args
    show_raw = "--raw" in args
    
    # Get tools and server mapping
    tools = context["tools"]
    tool_to_server_map = context.get("tool_to_server_map", {})
    
    # If no server mapping exists in context, build one
    if not tool_to_server_map:
        tool_to_server_map = {}
        for server_info in context["server_info"]:
            server_id = server_info["id"]
            server_name = server_info["name"]
            
            # Calculate tool ranges for each server
            start_idx = 0
            for prev_server in context["server_info"]:
                if prev_server["id"] < server_id:
                    start_idx += prev_server.get("tools", 0)
            
            end_idx = start_idx + server_info.get("tools", 0)
            
            # Associate tools with this server
            for i in range(start_idx, end_idx):
                if i < len(tools):
                    tool_name = tools[i]["name"]
                    tool_to_server_map[tool_name] = server_name
    
    if show_raw:
        # Show raw tool definitions (useful for debugging)
        from rich.syntax import Syntax
        import json
        
        raw_json = json.dumps(tools, indent=2)
        console.print(Syntax(raw_json, "json", theme="monokai", line_numbers=True))
        return True
    
    # Create a rich table
    table = Table(title=f"{len(tools)} Available Tools")
    
    # Add columns
    table.add_column("Server", style="cyan")
    table.add_column("Tool", style="green")
    table.add_column("Description")
    
    if show_all:
        # Show additional columns for detailed view
        table.add_column("Parameters", style="yellow")
    
    # Add rows for each tool
    for tool in tools:
        tool_name = tool["name"]
        server_name = tool_to_server_map.get(tool_name, "Unknown")
        
        # Truncate descriptions based on mode
        description = tool.get("description", "No description")
        if not show_all and len(description) > 75:
            description = description[:72] + "..."
        
        if show_all:
            # Get parameters information - check both "parameters" (OpenAI format) and "inputSchema" (MCP format)
            parameters = tool.get("parameters", tool.get("inputSchema", {}))
            
            # Handle different schema formats
            # OpenAI format
            if "properties" in parameters:
                properties = parameters.get("properties", {})
                required = parameters.get("required", [])
            # MCP format with inputSchema  
            elif parameters and isinstance(parameters, dict):
                properties = parameters.get("properties", {})
                required = parameters.get("required", [])
            else:
                properties = {}
                required = []
            
            # Format parameters as a string
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
    
    # Show parameter legend for detailed view
    if show_all:
        console.print("[yellow]* Required parameter[/yellow]")
    
    return True


# Register the command with completions
register_command("/tools", tools_command, ["--all", "--raw"])