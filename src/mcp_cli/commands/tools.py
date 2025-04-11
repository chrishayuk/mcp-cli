# mcp_cli/commands/tools.py
"""
Tools command module for listing and calling tools.
"""
import typer
import json
import asyncio
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

# app
app = typer.Typer(help="Tools commands")

@app.command("list")
async def tools_list(stream_manager, server_names=None):
    """
    List all tools from all servers.
    
    Args:
        stream_manager: StreamManager instance
        server_names: Optional dictionary mapping server indices to their names
    """
    print("[cyan]\nFetching Tools List from all servers...[/cyan]")
    
    all_tools = stream_manager.get_all_tools()
    server_info = stream_manager.get_server_info()
    
    if not all_tools:
        print("[yellow]No tools available from any server.[/yellow]")
        return
    
    # Group tools by server
    tools_by_server = {}
    for tool in all_tools:
        name = tool.get("name", "Unknown")
        server_name = stream_manager.get_server_for_tool(name)
        if server_name not in tools_by_server:
            tools_by_server[server_name] = []
        tools_by_server[server_name].append(tool)
    
    # Display tools for each server
    for server_name, tools in tools_by_server.items():
        # Create a table for this server's tools
        table = Table(title=f"{server_name} Tools ({len(tools)} available)")
        table.add_column("Tool", style="green")
        table.add_column("Description")
        
        for tool in tools:
            name = tool.get("name", "Unknown")
            desc = tool.get("description", "No description")
            if len(desc) > 75:
                desc = desc[:72] + "..."
            table.add_row(name, desc)
        
        print(table)
    
    # Print summary
    print(f"[green]Total tools available: {len(all_tools)}[/green]")

@app.command("call")
async def tools_call(stream_manager, server_names=None):
    """
    Call a tool with arguments.
    
    Args:
        stream_manager: StreamManager instance
        server_names: Optional dictionary mapping server indices to their names
    """
    print("[cyan]\nTool Call Interface[/cyan]")
    
    # Get all available tools
    all_tools = stream_manager.get_all_tools()
    
    if not all_tools:
        print("[yellow]No tools available from any server.[/yellow]")
        return
    
    # List available tools
    print("[green]Available tools:[/green]")
    for i, tool in enumerate(all_tools):
        server_name = stream_manager.get_server_for_tool(tool['name'])
        print(f"  {i+1}. {tool['name']} (from {server_name}) - {tool.get('description', 'No description')}")
    
    # Get user selection
    try:
        selection = int(input("\nEnter tool number to call: ")) - 1
        if not 0 <= selection < len(all_tools):
            print("[red]Invalid selection.[/red]")
            return
    except ValueError:
        print("[red]Please enter a number.[/red]")
        return
    
    # Get the selected tool
    tool = all_tools[selection]
    server_name = stream_manager.get_server_for_tool(tool['name'])
    
    # Show tool details
    print(f"\n[green]Selected: {tool['name']} from {server_name}[/green]")
    print(f"[cyan]Description: {tool.get('description', 'No description')}[/cyan]")
    
    # Show parameters
    parameters = tool.get("parameters", tool.get("inputSchema", {}))
    
    if "properties" in parameters:
        properties = parameters["properties"]
        required = parameters.get("required", [])
    
        if properties:
            print("\n[yellow]Parameters:[/yellow]")
            for name, details in properties.items():
                req = "[Required]" if name in required else "[Optional]"
                param_type = details.get("type", "any")
                description = details.get("description", "")
                print(f"  - {name} ({param_type}) {req}: {description}")
    
    # Get arguments from user
    print("\n[yellow]Enter arguments as JSON (empty for none):[/yellow]")
    args_input = input("> ")
    
    # Parse arguments
    if args_input.strip():
        try:
            args = json.loads(args_input)
        except json.JSONDecodeError:
            print("[red]Invalid JSON. Please try again.[/red]")
            return
    else:
        args = {}
    
    # Call the tool
    print(f"\n[cyan]Calling tool '{tool['name']}' from {server_name} with arguments:[/cyan]")
    print(Syntax(json.dumps(args, indent=2), "json"))
    
    try:
        # Use StreamManager to call the tool
        response = await stream_manager.call_tool(
            tool_name=tool['name'],
            arguments=args,
            server_name=server_name
        )
        
        # Check for errors
        if response.get("isError"):
            print(f"[red]Error calling tool: {response.get('error', 'Unknown error')}[/red]")
            return
        
        # Format and display results
        content = response.get("content", "No content returned")
        
        if isinstance(content, list) and content and isinstance(content[0], dict):
            # It's a list of objects, format as JSON
            print(f"\n[green]Tool response:[/green]")
            print(Syntax(json.dumps(content, indent=2), "json"))
        elif isinstance(content, dict):
            # It's a single object, format as JSON
            print(f"\n[green]Tool response:[/green]")
            print(Syntax(json.dumps(content, indent=2), "json"))
        else:
            # It's a simple value, just print it
            print(f"\n[green]Tool response:[/green] {content}")
        
    except Exception as e:
        print(f"[red]Error: {e}[/red]")