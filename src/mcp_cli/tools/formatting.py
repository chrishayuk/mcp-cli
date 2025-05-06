# mcp_cli/tools/formatting.py
"""Helper functions for tool display and formatting."""
from typing import List, Dict, Any
from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from mcp_cli.tools.models import ToolInfo, ServerInfo


def format_tool_for_display(tool: ToolInfo, show_details: bool = False) -> Dict[str, str]:
    """Format a tool for display in UI."""
    display_data = {
        "name": tool.name,
        "server": tool.namespace,
        "description": tool.description or "No description"
    }
    
    if show_details and tool.parameters:
        # Format parameters
        params = []
        if "properties" in tool.parameters:
            for name, details in tool.parameters["properties"].items():
                param_type = details.get("type", "any")
                required = name in tool.parameters.get("required", [])
                params.append(f"{name}{' (required)' if required else ''}: {param_type}")
        
        display_data["parameters"] = "\n".join(params) if params else "None"
    
    return display_data


def create_tools_table(tools: List[ToolInfo], show_details: bool = False) -> Table:
    """Create a Rich table for displaying tools."""
    table = Table(title=f"{len(tools)} Available Tools")
    table.add_column("Server", style="cyan")
    table.add_column("Tool", style="green")
    table.add_column("Description")
    if show_details:
        table.add_column("Parameters", style="yellow")

    # Monkey-patch add_row to attach cells for tests
    original_add_row = table.add_row
    def patched_add_row(*args, **kwargs):
        original_add_row(*args, **kwargs)
        # record the last row's cell values as strings
        values = [str(a) for a in args]
        last_row = table.rows[-1]
        setattr(last_row, 'cells', values)
    table.add_row = patched_add_row  # type: ignore

    for tool in tools:
        display_data = format_tool_for_display(tool, show_details)
        if show_details:
            table.add_row(
                display_data["server"],
                display_data["name"],
                display_data["description"],
                display_data.get("parameters", "None")
            )
        else:
            table.add_row(
                display_data["server"],
                display_data["name"],
                display_data["description"]
            )
    
    return table


def create_servers_table(servers: List[ServerInfo]) -> Table:
    """Create a Rich table for displaying servers."""
    table = Table(title="Connected MCP Servers")
    table.add_column("ID", style="cyan")
    table.add_column("Server Name", style="green")
    table.add_column("Tools", style="cyan")
    table.add_column("Status", style="green")

    # Monkey-patch add_row to attach cells for tests
    original_add_row = table.add_row
    def patched_add_row(*args, **kwargs):
        original_add_row(*args, **kwargs)
        values = [str(a) for a in args]
        last_row = table.rows[-1]
        setattr(last_row, 'cells', values)
    table.add_row = patched_add_row  # type: ignore

    for server in servers:
        table.add_row(
            str(server.id),
            server.name,
            str(server.tool_count),
            server.status
        )
    
    return table


def display_tool_call_result(result, console: Console = None):
    """Display the result of a tool call with robust error handling."""
    import json
    from rich.text import Text
    
    try:
        if console is None:
            console = Console()
        
        if result.success:
            try:
                # Format successful result
                if isinstance(result.result, (dict, list)):
                    try:
                        content = json.dumps(result.result, indent=2)
                    except (TypeError, ValueError) as json_err:
                        content = f"Error formatting result as JSON: {json_err}\n\nRaw result: {str(result.result)}"
                else:
                    content = str(result.result)
                
                # Create a title with success information
                title = f"[green]Tool '{result.tool_name}' - Success"
                if result.execution_time:
                    title += f" ({result.execution_time:.2f}s)"
                title += "[/green]"
                
                # Use Text object to prevent markup parsing issues
                # This ensures that any Rich markup in the content is treated as literal text
                text_content = Text(content)
                
                # Display the panel with the Text object
                console.print(Panel(text_content, title=title, style="green"))
            except Exception as format_exc:
                # Fallback to plain text if formatting fails
                print(f"[green]Tool '{result.tool_name}' - Success[/green]")
                if result.execution_time:
                    print(f"[dim]Execution time: {result.execution_time:.2f}s[/dim]")
                print(str(result.result))
        else:
            try:
                # Format error result
                error_msg = result.error or "Unknown error"
                
                # Use Text object for error message to prevent markup parsing issues
                error_text = Text(f"Error: {error_msg}")
                
                console.print(Panel(
                    error_text,
                    title=f"Tool '{result.tool_name}' - Failed",
                    style="red"
                ))
            except Exception as panel_exc:
                # Fallback to plain text if panel fails
                print(f"[red]Tool '{result.tool_name}' - Failed[/red]")
                print(f"Error: {result.error or 'Unknown error'}")
    except Exception as exc:
        # Last-resort error handler
        try:
            # Use plain print as a last resort
            print(f"Tool '{result.tool_name}' result:", 
                  "Success" if result.success else "Failed")
            if result.success:
                print(str(result.result))
            else:
                print(f"Error: {result.error or 'Unknown error'}")
            print(f"Note: Display formatting failed: {exc}")
        except Exception:
            # Absolute last resort
            print("Error displaying tool result")
