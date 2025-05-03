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

# Import formatting helpers
from mcp_cli.tools.formatting import create_tools_table, display_tool_call_result

# app
app = typer.Typer(help="Tools commands")

@app.command("list")
async def tools_list(tool_manager, server_names=None):
    """
    List all tools from all servers.

    Args:
        tool_manager: ToolManager instance
        server_names: Optional dictionary mapping server indices to their names
    """
    print("[cyan]\nFetching Tools List from all servers...[/cyan]")
    
    all_tools = tool_manager.get_all_tools()
    if not all_tools:
        print("[yellow]No tools available from any server.[/yellow]")
        return
    
    # Group tools by server
    tools_by_server = {}
    for tool in all_tools:
        server_name = tool.namespace  # Use namespace as server name
        tools_by_server.setdefault(server_name, []).append(tool)
    
    # Display tools by server
    for srv, tools in tools_by_server.items():
        table = Table(title=f"{srv} Tools ({len(tools)} available)")
        table.add_column("Tool", style="green")
        table.add_column("Description")
        for tool in tools:
            desc = tool.description or "No description"
            if len(desc) > 75:
                desc = desc[:72] + "..."
            table.add_row(tool.name, desc)
        print(table)
    
    print(f"[green]Total tools available: {len(all_tools)}[/green]")

@app.command("call")
async def tools_call(tool_manager, server_names=None):
    """
    Call a tool with arguments.

    Args:
        tool_manager: ToolManager instance
        server_names: Optional dictionary mapping server indices to their names
    """
    print("[cyan]\nTool Call Interface[/cyan]")
    
    all_tools = tool_manager.get_all_tools()
    if not all_tools:
        print("[yellow]No tools available from any server.[/yellow]")
        return

    # List available tools
    print("[green]Available tools:[/green]")
    for idx, tool in enumerate(all_tools, start=1):
        print(f"  {idx}. {tool.name} (from {tool.namespace}) - {tool.description or 'No description'}")
    
    # Get user selection asynchronously
    sel_raw = await asyncio.to_thread(input, "\nEnter tool number to call: ")
    try:
        selection = int(sel_raw) - 1
        if not (0 <= selection < len(all_tools)):
            print("[red]Invalid selection.[/red]")
            return
    except ValueError:
        print("[red]Please enter a valid number.[/red]")
        return

    tool = all_tools[selection]

    print(f"\n[green]Selected: {tool.name} from {tool.namespace}[/green]")
    print(f"[cyan]Description: {tool.description or 'No description'}[/cyan]")

    # Show parameters
    params = tool.parameters or {}
    if isinstance(params, dict) and 'properties' in params:
        props = params['properties']
        required = params.get('required', [])
        if props:
            print("\n[yellow]Parameters:[/yellow]")
            for name, details in props.items():
                req = '[Required]' if name in required else '[Optional]'
                ptype = details.get('type', 'any')
                desc = details.get('description', '')
                print(f"  - {name} ({ptype}) {req}: {desc}")

    # Get arguments asynchronously
    print("\n[yellow]Enter arguments as JSON (empty for none):[/yellow]")
    args_input = await asyncio.to_thread(input, "> ")
    if args_input.strip():
        try:
            args = json.loads(args_input)
        except json.JSONDecodeError:
            print("[red]Invalid JSON. Please try again.[/red]")
            return
    else:
        args = {}

    print(f"\n[cyan]Calling tool '{tool.name}' from {tool.namespace} with arguments:[/cyan]")
    print(Syntax(json.dumps(args, indent=2), 'json'))

    try:
        # Use ToolManager to execute the tool
        result = await tool_manager.execute_tool(tool.name, args)
        
        # Use the formatting helper to display the result nicely
        display_tool_call_result(result)
        
    except Exception as e:
        print(f"[red]Error: {e}[/red]")