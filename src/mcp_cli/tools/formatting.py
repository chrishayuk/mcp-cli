# mcp_cli/tools/formatting.py
"""Helper functions for tool display and formatting."""

from typing import List, Dict
from rich.table import Table
from chuk_term.ui import output
from rich.panel import Panel

from mcp_cli.tools.models import ToolInfo, ServerInfo


def format_tool_for_display(
    tool: ToolInfo, show_details: bool = False
) -> Dict[str, str]:
    """Format a tool for display in UI."""
    display_data = {
        "name": tool.name,
        "server": tool.namespace,
        "description": tool.description or "No description",
    }

    if show_details and tool.parameters:
        # Format parameters
        params = []
        if "properties" in tool.parameters:
            for name, details in tool.parameters["properties"].items():
                param_type = details.get("type", "any")
                required = name in tool.parameters.get("required", [])
                params.append(
                    f"{name}{' (required)' if required else ''}: {param_type}"
                )

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
        setattr(last_row, "cells", values)

    table.add_row = patched_add_row  # type: ignore

    for tool in tools:
        display_data = format_tool_for_display(tool, show_details)
        if show_details:
            table.add_row(
                display_data["server"],
                display_data["name"],
                display_data["description"],
                display_data.get("parameters", "None"),
            )
        else:
            table.add_row(
                display_data["server"],
                display_data["name"],
                display_data["description"],
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
        setattr(last_row, "cells", values)

    table.add_row = patched_add_row  # type: ignore

    for server in servers:
        table.add_row(
            str(server.id), server.name, str(server.tool_count), server.status
        )

    return table


def format_content_with_ansi_support(content: str, no_wrap: bool = False):
    """
    Helper function to format content with proper ANSI escape sequence handling.
    
    Args:
        content: The content to format
        no_wrap: If True, disable wrapping (for ASCII art/charts)
    
    Returns:
        Rich Text object with proper formatting
    """
    from rich.text import Text
    
    # Check if content looks like ASCII art/chart (contains box drawing characters or ANSI codes)
    box_chars = {'┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '│', '─', '█', '▄', '▀', '▌', '▐'}
    has_ansi = '\033[' in content or '\x1b[' in content  # Check for ANSI escape sequences
    is_ascii_art = any(char in content for char in box_chars) or has_ansi
    
    if is_ascii_art or no_wrap:
        # For ASCII art, need special handling to preserve formatting
        # First, clean up any invisible characters that might interfere
        clean_content = content.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')  # Remove zero-width spaces
        
        # Handle ANSI content by using Text.from_ansi() to properly interpret escape sequences
        if has_ansi:
            return Text.from_ansi(clean_content, no_wrap=True)
        else:
            return Text(clean_content, no_wrap=True, overflow="ignore")
    else:
        # For regular content, use Text object with normal wrapping behavior
        # Use "ellipsis" instead of "fold" to ensure proper line breaking
        return Text(content)


def display_tool_call_result(result, console=None):
    """Display the result of a tool call."""
    import json
    from rich.text import Text

    # If console is provided, use it; otherwise use output.print
    print_func = console.print if console else output.print

    if result.success:
        # Format successful result
        if isinstance(result.result, (dict, list)):
            try:
                content = json.dumps(result.result, indent=2)
            except Exception:
                content = str(result.result)
        else:
            content = str(result.result)

        title = f"[green]Tool '{result.tool_name}' - Success"
        if result.execution_time:
            title += f" ({result.execution_time:.2f}s)"
        title += "[/green]"

        # Check if we need special handling for ASCII art
        box_chars = {'┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '│', '─', '█', '▄', '▀', '▌', '▐'}
        has_ansi = '\033[' in content or '\x1b[' in content
        is_ascii_art = any(char in content for char in box_chars) or has_ansi
        
        if is_ascii_art:
            # For ASCII art, use the helper function and optimized panel settings
            text_content = format_content_with_ansi_support(content, no_wrap=True)
            print_func(Panel(
                text_content,
                title=title, 
                style="green", 
                width=None, 
                padding=(0, 0),
                expand=True,
                safe_box=False
            ))
        else:
            # For regular content, force proper text wrapping without ellipses
            from rich.text import Text
            # Create Text object with forced wrapping - NO ellipses mode
            text_content = Text(content, overflow="fold", no_wrap=False)
            print_func(Panel(
                text_content, 
                title=title, 
                style="green", 
                expand=False,
                width=None,  # Let it size naturally
                padding=(0, 1)  # Small padding for readability
            ))
    else:
        # Format error result
        error_msg = result.error or "Unknown error"

        # Use Text object for error message too
        error_text = Text(f"Error: {error_msg}")

        print_func(
            Panel(error_text, title=f"Tool '{result.tool_name}' - Failed", style="red")
        )
