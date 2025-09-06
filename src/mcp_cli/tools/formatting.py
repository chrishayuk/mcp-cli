# mcp_cli/tools/formatting.py
"""Helper functions for tool display and formatting using chuk-term."""

from typing import List, Dict
from chuk_term.ui import output

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


def create_tools_table(tools: List[ToolInfo], show_details: bool = False):
    """Create a Rich table for tools (does not print it)."""
    # Prepare data for table
    headers = ["Server", "Tool", "Description"]
    if show_details:
        headers.append("Parameters")

    rows = []
    for tool in tools:
        display_data = format_tool_for_display(tool, show_details)
        row = [
            display_data["server"],
            display_data["name"],
            display_data["description"],
        ]
        if show_details:
            row.append(display_data.get("parameters", "None"))
        rows.append(row)

    # Create a Rich table
    from rich.table import Table

    table = Table(title=f"{len(tools)} Available Tools")
    for header in headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*row)

    # Return the Rich table object
    return table


def create_servers_table(servers: List[ServerInfo]):
    """Create a Rich table for servers (does not print it)."""
    # Prepare data for table
    headers = ["ID", "Server Name", "Tools", "Status"]
    rows = []

    for server in servers:
        rows.append(
            [str(server.id), server.name, str(server.tool_count), server.status]
        )

    # Create a Rich table
    from rich.table import Table

    table = Table(title="Connected MCP Servers")
    for header in headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*row)

    # Return the Rich table object
    return table


def display_tool_call_result(result, console=None):
    """Display the result of a tool call using chuk-term."""
    import json

    if result.success:
        # Display success header with timing
        title = f"✓ Tool '{result.tool_name}' completed"
        if result.execution_time:
            title += f" ({result.execution_time:.2f}s)"
        output.success(title)

        # Format and display the result based on type
        if isinstance(result.result, list) and result.result:
            # For lists, check if they're records (list of dicts)
            if all(isinstance(item, dict) for item in result.result):
                # Display as a table if it's a list of records
                if len(result.result) <= 10:  # Show table for small result sets
                    headers = list(result.result[0].keys()) if result.result else []
                    rows = [
                        [str(item.get(h, "")) for h in headers]
                        for item in result.result
                    ]
                    from rich.table import Table

                    table = Table()
                    for header in headers:
                        table.add_column(header)
                    for row in rows:
                        table.add_row(*[str(cell) for cell in row])
                    output.print_table(table)
                else:
                    # For large result sets, show summary
                    output.info(f"Returned {len(result.result)} records")
                    # Show first few items
                    output.print("First 3 records:")
                    for i, item in enumerate(result.result[:3], 1):
                        output.print(
                            f"  {i}. {json.dumps(item, separators=(',', ': '))}"
                        )
                    if len(result.result) > 3:
                        output.print(f"  ... and {len(result.result) - 3} more")
            else:
                # For simple lists, show items
                output.info(f"Returned {len(result.result)} items")
                for item in result.result[:10]:
                    output.print(f"  • {item}")
                if len(result.result) > 10:
                    output.print(f"  ... and {len(result.result) - 10} more")
        elif isinstance(result.result, dict):
            # For dictionaries, show as key-value pairs if small
            if len(result.result) <= 10:
                output.kvpairs(result.result)
            else:
                # For large dicts, show as formatted JSON
                content = json.dumps(result.result, indent=2)
                if len(content) > 500:
                    # Truncate very large results
                    output.code(content[:500] + "\n... (truncated)", language="json")
                else:
                    output.code(content, language="json")
        elif isinstance(result.result, str):
            # For strings, display them nicely
            if len(result.result) > 500:
                output.print(result.result[:500] + "... (truncated)")
            else:
                output.print(result.result)
        else:
            # For other types, format as JSON
            try:
                import json

                formatted = json.dumps(result.result, indent=2)
                if len(formatted) > 500:
                    output.code(formatted[:500] + "\n... (truncated)", language="json")
                else:
                    output.code(formatted, language="json")
            except:
                # Fallback to string representation
                output.print(str(result.result))
    else:
        # Display error
        error_msg = result.error or "Unknown error"
        output.error(f"✗ Tool '{result.tool_name}' failed")
        output.print(f"Error: {error_msg}")
