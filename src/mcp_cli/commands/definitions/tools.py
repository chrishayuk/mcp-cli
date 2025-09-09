# src/mcp_cli/commands/definitions/tools.py
"""
Unified tools command implementation.
"""

from __future__ import annotations

import json
from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandParameter,
    CommandResult,
)
from mcp_cli.context import get_context
from chuk_term.ui import output, format_table


class ToolsCommand(UnifiedCommand):
    """List and inspect MCP tools."""

    @property
    def name(self) -> str:
        return "tools"

    @property
    def aliases(self) -> List[str]:
        return []

    @property
    def description(self) -> str:
        return "List and inspect MCP tools"

    @property
    def help_text(self) -> str:
        return """
List and inspect MCP tools from connected servers.

Usage:
  /tools                - List all tools (with truncated descriptions)
  /tools <server_name>  - List tools from a specific server
  /tools <tool_name>    - Show detailed information about a specific tool

Options:
  --raw                 - Output as JSON
  --details             - Show full descriptions and parameters

Examples:
  /tools                - Show all tools in a table
  /tools sqlite         - Show only tools from the sqlite server
  /tools echo_text      - Show detailed info about the echo_text tool
  /tools --raw          - Output all tools as JSON
"""

    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="filter",
                type=str,
                required=False,
                help="Server name to filter by, or tool name for details",
            ),
            CommandParameter(
                name="raw",
                type=bool,
                default=False,
                help="Output as JSON",
                is_flag=True,
            ),
            CommandParameter(
                name="details",
                type=bool,
                default=False,
                help="Show full descriptions",
                is_flag=True,
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the tools command."""
        # Get context and tool manager
        context = get_context()
        if not context or not context.tool_manager:
            return CommandResult(
                success=False,
                error="No tool manager available. Please connect to a server first.",
            )

        tm = context.tool_manager

        # Get parameters
        filter_arg = kwargs.get("filter")
        if not filter_arg and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list) and args_val:
                filter_arg = args_val[0]
            elif isinstance(args_val, str):
                filter_arg = args_val

        show_raw = kwargs.get("raw", False)
        show_full_desc = kwargs.get("details", False)

        try:
            # Get all tools
            all_tools = await tm.get_unique_tools()

            if not all_tools:
                return CommandResult(
                    success=True,
                    output="No tools available.",
                )

            # If filter provided, check if it's a server or tool name
            if filter_arg:
                # First check if it's a tool name (exact match)
                tool_match = None
                for tool in all_tools:
                    if (
                        tool.name == filter_arg
                        or tool.fully_qualified_name == filter_arg
                    ):
                        tool_match = tool
                        break

                if tool_match:
                    # Show detailed tool information
                    return self._show_tool_details(tool_match)

                # Helper to get actual server name (reuse the function)
                def get_server_name(tool):
                    tool_server_map = {
                        "read_query": "sqlite",
                        "write_query": "sqlite",
                        "create_table": "sqlite",
                        "list_tables": "sqlite",
                        "describe_table": "sqlite",
                        "append_insight": "sqlite",
                        "echo_text": "echo",
                        "echo_uppercase": "echo",
                        "echo_reverse": "echo",
                        "echo_json": "echo",
                        "echo_list": "echo",
                        "echo_number": "echo",
                        "echo_delay": "echo",
                        "echo_error": "echo",
                        "get_service_info": "echo",
                    }
                    if tool.name in tool_server_map:
                        return tool_server_map[tool.name]
                    elif tool.namespace and tool.namespace != "stdio":
                        return tool.namespace
                    elif "echo" in tool.name.lower():
                        return "echo"
                    elif any(
                        x in tool.name.lower()
                        for x in ["query", "table", "sql", "database"]
                    ):
                        return "sqlite"
                    return tool.namespace

                # Check if it's a server name
                server_tools = [
                    t for t in all_tools if get_server_name(t) == filter_arg
                ]
                if server_tools:
                    # Show tools from this server
                    return self._show_tools_table(
                        server_tools,
                        show_raw,
                        show_full_desc,
                        f"Tools from '{filter_arg}' server",
                    )

                # No match found
                return CommandResult(
                    success=False,
                    error=f"No tool or server found matching '{filter_arg}'",
                )

            # Show all tools
            return self._show_tools_table(
                all_tools, show_raw, show_full_desc, "Available Tools"
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list tools: {str(e)}",
            )

    def _show_tool_details(self, tool) -> CommandResult:
        """Show detailed information about a specific tool."""

        # Helper to get actual server name
        def get_server_name(tool):
            """Map tool to its actual server name based on tool patterns."""
            tool_server_map = {
                # SQLite tools
                "read_query": "sqlite",
                "write_query": "sqlite",
                "create_table": "sqlite",
                "list_tables": "sqlite",
                "describe_table": "sqlite",
                "append_insight": "sqlite",
                # Echo tools
                "echo_text": "echo",
                "echo_uppercase": "echo",
                "echo_reverse": "echo",
                "echo_json": "echo",
                "echo_list": "echo",
                "echo_number": "echo",
                "echo_delay": "echo",
                "echo_error": "echo",
                "get_service_info": "echo",
            }

            if tool.name in tool_server_map:
                return tool_server_map[tool.name]
            elif tool.namespace and tool.namespace != "stdio":
                return tool.namespace
            elif "echo" in tool.name.lower():
                return "echo"
            elif any(
                x in tool.name.lower() for x in ["query", "table", "sql", "database"]
            ):
                return "sqlite"
            return tool.namespace

        server_name = get_server_name(tool)

        # Format tool details
        details = []
        details.append(f"Tool: {tool.name}")
        details.append(f"Server: {server_name}")
        details.append(f"Full Name: {server_name}.{tool.name}")
        details.append("")
        details.append("Description:")
        details.append(tool.description or "No description available")

        if tool.parameters:
            details.append("")
            details.append("Parameters:")

            if "properties" in tool.parameters:
                props = tool.parameters["properties"]
                required = tool.parameters.get("required", [])

                for param_name, param_details in props.items():
                    param_type = param_details.get("type", "any")
                    param_desc = param_details.get("description", "")
                    is_required = param_name in required
                    default = param_details.get("default")

                    details.append(f"  - {param_name}:")
                    details.append(f"      Type: {param_type}")
                    if param_desc:
                        details.append(f"      Description: {param_desc}")
                    details.append(f"      Required: {'Yes' if is_required else 'No'}")
                    if default is not None:
                        details.append(f"      Default: {default}")
            else:
                details.append("  No parameters defined")
        else:
            details.append("")
            details.append("Parameters: None")

        # Display in a panel
        output.panel(
            "\n".join(details),
            title=f"Tool Details: {tool.name}",
            style="cyan",
        )

        return CommandResult(success=True)

    def _show_tools_table(
        self, tools: List, show_raw: bool, show_full_desc: bool, title: str
    ) -> CommandResult:
        """Show tools in a table format."""
        if show_raw:
            # Output as JSON
            tools_data = []
            for tool in tools:
                tools_data.append(
                    {
                        "name": tool.name,
                        "server": tool.namespace,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                )
            output.print(json.dumps(tools_data, indent=2))
            return CommandResult(success=True, data=tools_data)

        # Build table data
        output.info("\nFetching tool catalogue from connected servers…")

        # Helper function to determine actual server name
        def get_server_name(tool):
            """Map tool to its actual server name based on tool patterns."""
            # Map known tool patterns to servers
            tool_server_map = {
                # SQLite tools
                "read_query": "sqlite",
                "write_query": "sqlite",
                "create_table": "sqlite",
                "list_tables": "sqlite",
                "describe_table": "sqlite",
                "append_insight": "sqlite",
                # Echo tools
                "echo_text": "echo",
                "echo_uppercase": "echo",
                "echo_reverse": "echo",
                "echo_json": "echo",
                "echo_list": "echo",
                "echo_number": "echo",
                "echo_delay": "echo",
                "echo_error": "echo",
                "get_service_info": "echo",
            }

            # Try to get server from mapping
            if tool.name in tool_server_map:
                return tool_server_map[tool.name]

            # Fall back to namespace if not in mapping
            # This handles HTTP servers and other transports correctly
            if tool.namespace and tool.namespace != "stdio":
                return tool.namespace

            # Last resort - try to guess from tool name patterns
            if "echo" in tool.name.lower():
                return "echo"
            elif any(
                x in tool.name.lower() for x in ["query", "table", "sql", "database"]
            ):
                return "sqlite"

            # Default to namespace
            return tool.namespace

        table_data = []
        for tool in tools:
            desc = tool.description or "No description"

            # Truncate description unless --details flag is used
            if not show_full_desc and len(desc) > 80:
                desc = desc[:77] + "..."

            # Get the actual server name
            server_name = get_server_name(tool)

            table_data.append(
                {
                    "Server": server_name,
                    "Tool": tool.name,
                    "Description": desc,
                }
            )

        # Display table
        table = format_table(
            table_data,
            title=f"{len(tools)} {title}",
            columns=["Server", "Tool", "Description"],
        )
        output.print_table(table)
        output.success(f"Total tools available: {len(tools)}")

        # Add appropriate tip based on context
        if "server" in title.lower():
            output.tip(
                "💡 Use: /tools <tool_name> for details  |  /tools --details for full descriptions"
            )
        else:
            output.tip(
                "💡 Use: /tools <server> to filter  |  /tools <tool_name> for details  |  /tools --details for full descriptions"
            )

        return CommandResult(success=True, data=table_data)
