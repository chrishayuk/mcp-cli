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
    if not all_tools:
        print("[yellow]No tools available from any server.[/yellow]")
        return
    
    # Group tools by server
    tools_by_server = {}
    for tool in all_tools:
        name = tool.get("name", "Unknown")
        srv = stream_manager.get_server_for_tool(name)
        tools_by_server.setdefault(srv, []).append(tool)
    
    # Display tools
    for srv, tools in tools_by_server.items():
        table = Table(title=f"{srv} Tools ({len(tools)} available)")
        table.add_column("Tool", style="green")
        table.add_column("Description")
        for tool in tools:
            desc = tool.get("description", "No description")
            if len(desc) > 75:
                desc = desc[:72] + "..."
            table.add_row(tool.get("name", "Unknown"), desc)
        print(table)
    
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
    
    all_tools = stream_manager.get_all_tools()
    if not all_tools:
        print("[yellow]No tools available from any server.[/yellow]")
        return

    # List available tools
    print("[green]Available tools:[/green]")
    for idx, tool in enumerate(all_tools, start=1):
        srv = stream_manager.get_server_for_tool(tool['name'])
        print(f"  {idx}. {tool['name']} (from {srv}) - {tool.get('description','No description')}")
    
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
    srv = stream_manager.get_server_for_tool(tool['name'])

    print(f"\n[green]Selected: {tool['name']} from {srv}[/green]")
    print(f"[cyan]Description: {tool.get('description','No description')}[/cyan]")

    # Show parameters
    params = tool.get('parameters', tool.get('inputSchema', {}))
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

    print(f"\n[cyan]Calling tool '{tool['name']}' from {srv} with arguments:[/cyan]")
    print(Syntax(json.dumps(args, indent=2), 'json'))

    try:
        response = await stream_manager.call_tool(
            tool_name=tool['name'],
            arguments=args,
            server_name=srv
        )
        if response.get('isError'):
            print(f"[red]Error calling tool: {response.get('error','Unknown error')}[/red]")
            return

        content = response.get('content', 'No content returned')
        if isinstance(content, list) and content and isinstance(content[0], dict):
            print("\n[green]Tool response:[/green]")
            print(Syntax(json.dumps(content, indent=2), 'json'))
        elif isinstance(content, dict):
            print("\n[green]Tool response:[/green]")
            print(Syntax(json.dumps(content, indent=2), 'json'))
        else:
            print(f"\n[green]Tool response:[/green] {content}")
    except Exception as e:
        print(f"[red]Error: {e}[/red]")
