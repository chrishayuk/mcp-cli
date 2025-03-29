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

# imports
from chuk_mcp.mcp_client.messages.tools.send_messages import send_tools_list, send_tools_call

# app
app = typer.Typer(help="Tools commands")

@app.command("list")
async def tools_list(server_streams: list, server_names=None):
    """
    List all tools from all servers.
    
    Args:
        server_streams: List of (read_stream, write_stream) tuples
        server_names: Optional dictionary mapping server indices to their names
    """
    print("[cyan]\nFetching Tools List from all servers...[/cyan]")
    
    all_tools = []
    tasks = []
    
    try:
        # Create tasks for fetching tools from all servers concurrently
        for i, (r_stream, w_stream) in enumerate(server_streams):
            # Get server name if available
            server_display_name = f"Server {i+1}"
            if server_names and i in server_names:
                server_display_name = server_names[i]
            
            task = asyncio.create_task(fetch_tools_from_server(
                i, r_stream, w_stream, server_display_name
            ))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and collect tools
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[yellow]Error fetching tools: {result}[/yellow]")
                continue
                
            server_name, tools, table = result
            if tools:
                all_tools.extend(tools)
                print(table)
            else:
                print(f"[yellow]{server_name}: No tools available.[/yellow]")
        
        # Summary
        if all_tools:
            print(f"[green]Total tools available: {len(all_tools)}[/green]")
        else:
            print("[yellow]No tools available from any server.[/yellow]")
    except Exception as e:
        print(f"[red]Error listing tools: {e}[/red]")
    finally:
        # Clean up tasks that might still be running
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Let any cancelled tasks complete their cancellation
        if tasks:
            await asyncio.wait(tasks, timeout=0.5)

async def fetch_tools_from_server(server_idx, r_stream, w_stream, server_display_name):
    """Fetch tools from a single server and format the results."""
    try:
        response = await send_tools_list(r_stream, w_stream)
        tools = response.get("tools", [])
        
        # Create a table for this server's tools
        table = Table(title=f"{server_display_name} Tools ({len(tools)} available)")
        table.add_column("Tool", style="green")
        table.add_column("Description")
        
        for tool in tools:
            name = tool.get("name", "Unknown")
            desc = tool.get("description", "No description")
            if len(desc) > 75:
                desc = desc[:72] + "..."
            table.add_row(name, desc)
        
        return server_display_name, tools, table
    except Exception as e:
        raise Exception(f"Failed to fetch tools from {server_display_name}: {e}")

@app.command("call")
async def tools_call(server_streams: list, server_names=None):
    """
    Call a tool with arguments.
    
    Args:
        server_streams: List of (read_stream, write_stream) tuples
        server_names: Optional dictionary mapping server indices to their names
    """
    print("[cyan]\nTool Call Interface[/cyan]")
    
    # First, get a list of all available tools
    all_tools = []
    try:
        for i, (r_stream, w_stream) in enumerate(server_streams):
            # Get server name if available
            server_display_name = f"Server {i+1}"
            if server_names and i in server_names:
                server_display_name = server_names[i]
            
            try:
                response = await send_tools_list(r_stream, w_stream)
                tools = response.get("tools", [])
                for tool in tools:
                    tool["server_index"] = i
                    tool["server_name"] = server_display_name
                all_tools.extend(tools)
            except Exception as e:
                print(f"[yellow]Error fetching tools from {server_display_name}: {e}[/yellow]")
        
        if not all_tools:
            print("[yellow]No tools available from any server.[/yellow]")
            return
        
        # List available tools
        print("[green]Available tools:[/green]")
        for i, tool in enumerate(all_tools):
            print(f"  {i+1}. {tool['name']} (from {tool['server_name']}) - {tool.get('description', 'No description')}")
        
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
        server_index = tool["server_index"]
        r_stream, w_stream = server_streams[server_index]
        
        # Show tool details
        print(f"\n[green]Selected: {tool['name']} from {tool['server_name']}[/green]")
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
        print(f"\n[cyan]Calling tool '{tool['name']}' from {tool['server_name']} with arguments:[/cyan]")
        print(Syntax(json.dumps(args, indent=2), "json"))
        
        try:
            response = await send_tools_call(
                read_stream=r_stream,
                write_stream=w_stream,
                name=tool["name"],
                arguments=args
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
    except Exception as e:
        print(f"[red]Error in tool call: {e}[/red]")
    finally:
        # Ensure any pending tasks are cleaned up
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Let any cancelled tasks complete their cancellation
        if tasks:
            await asyncio.wait(tasks, timeout=0.5)