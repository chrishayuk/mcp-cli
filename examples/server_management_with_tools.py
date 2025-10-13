#!/usr/bin/env python
"""
Complete end-to-end example of MCP server management with tool execution.

This script demonstrates:
1. Adding a new MCP server
2. Reloading to connect to the server
3. Executing tools from the newly added server
4. Removing the server

This example uses the time server which provides simple, safe tools to demonstrate.
"""

import asyncio
import json

# Initialize context before imports that need it
from mcp_cli.context import initialize_context, get_context

initialize_context()

from mcp_cli.commands.actions.servers import servers_action_async  # noqa: E402
from mcp_cli.config.config_manager import ConfigManager  # noqa: E402
from chuk_term.ui import output  # noqa: E402


async def add_time_server():
    """Add the MCP time server."""
    output.rule("[bold cyan]Adding MCP Time Server[/bold cyan]")

    # Add time server using uvx
    await servers_action_async(
        args=["add", "demo-time", "stdio", "uvx", "mcp-server-time"]
    )

    output.success("✅ Time server added to configuration")
    print()


async def connect_to_servers():
    """Initialize tool manager with configured servers."""
    output.rule("[bold cyan]Connecting to Servers[/bold cyan]")

    # Re-initialize context with the new server configuration
    config_manager = ConfigManager()
    config = config_manager.reload()

    # Get context and reinitialize tool manager
    context = get_context()

    # Import here to avoid circular dependency
    from mcp_cli.tools.manager import ToolManager

    # Create new tool manager with updated config
    enabled_servers = config.list_enabled_servers()
    output.info(f"Loading {len(enabled_servers)} configured server(s)...")

    # Convert ServerConfig to format expected by ToolManager
    server_configs = []
    for server in enabled_servers:
        if server.command:
            server_configs.append(
                {
                    "name": server.name,
                    "command": server.command,
                    "args": server.args,
                    "env": server.env,
                }
            )

    # Create and initialize tool manager
    tm = ToolManager(server_configs)
    await tm.initialize()

    # Update context with new tool manager
    context.tool_manager = tm

    output.success(f"✅ Connected to {len(server_configs)} server(s)")

    # Show connected servers
    servers = await tm.get_server_info() if hasattr(tm, "get_server_info") else []
    for server in servers:
        output.info(f"  • {server.name}: {server.tool_count} tools")

    print()
    return tm


async def execute_time_tools(tm):
    """Execute tools from the time server."""
    output.rule("[bold green]Executing Time Server Tools[/bold green]")

    if not tm:
        output.error("Tool manager not available")
        return

    # Get available tools
    tools = await tm.get_tools()

    # Find time server tools
    time_tools = [
        t
        for t in tools
        if "time" in t.name.lower() or "demo-time" in str(getattr(t, "namespace", ""))
    ]

    if not time_tools:
        output.warning("No time tools found. Showing all available tools:")
        for tool in tools[:5]:  # Show first 5 tools
            output.info(f"  • {tool.name}")
        print()

        # Try to execute a simple tool
        if tools:
            tool = tools[0]
            output.info(f"Executing first available tool: {tool.name}")

            try:
                # Build tool request
                tool_request = {"name": tool.name, "arguments": {}}

                # Execute tool
                result = await tm.execute_tool(tool_request)

                if result:
                    output.success("✅ Tool executed successfully!")
                    if isinstance(result, dict):
                        output.print(
                            json.dumps(result, indent=2)[:500]
                        )  # Truncate long output
                    else:
                        output.print(str(result)[:500])
                else:
                    output.warning("Tool returned no result")

            except Exception as e:
                output.error(f"Error executing tool: {e}")
    else:
        # Execute time tools
        output.info(f"Found {len(time_tools)} time-related tool(s)")

        for tool in time_tools[:2]:  # Execute first 2 tools
            output.info(f"\nExecuting: {tool.name}")

            try:
                # Build tool request
                tool_request = {"name": tool.name, "arguments": {}}

                # Execute tool
                result = await tm.execute_tool(tool_request)

                if result:
                    output.success(f"✅ {tool.name} executed!")
                    if isinstance(result, dict):
                        output.print(json.dumps(result, indent=2)[:300])
                    else:
                        output.print(str(result)[:300])
                else:
                    output.warning(f"{tool.name} returned no result")

            except Exception as e:
                output.error(f"Error executing {tool.name}: {e}")

    print()


async def remove_time_server():
    """Remove the time server."""
    output.rule("[bold cyan]Removing Time Server[/bold cyan]")

    await servers_action_async(args=["remove", "demo-time"])

    output.success("✅ Time server removed from configuration")
    print()


async def main():
    """Run the complete server management demo with tool execution."""

    output.rule(
        "[bold magenta]🚀 MCP Server Management with Tool Execution[/bold magenta]"
    )
    output.print()

    try:
        # Initialize context
        initialize_context()

        # Step 1: Show initial servers
        output.rule("[bold]Step 1: Initial Servers[/bold]")
        await servers_action_async(args=["list"])
        print()

        # Step 2: Add time server
        await add_time_server()

        # Step 3: Show servers after adding
        output.rule("[bold]Step 3: Servers After Adding[/bold]")
        await servers_action_async(args=["list"])
        print()

        # Step 4: Connect to servers (including new one)
        tm = await connect_to_servers()

        # Step 5: Execute tools from the time server
        if tm:
            await execute_time_tools(tm)

            # Cleanup: close tool manager
            await tm.cleanup()

        # Step 6: Remove the time server
        await remove_time_server()

        # Step 7: Show final servers
        output.rule("[bold]Step 7: Final Servers[/bold]")
        await servers_action_async(args=["list"])
        print()

        # Summary
        output.rule("[bold green]✅ Demo Complete![/bold green]")
        output.success("Successfully demonstrated full server lifecycle:")
        output.info("  1. Added time server to configuration")
        output.info("  2. Connected to the server")
        output.info("  3. Executed tools from the server")
        output.info("  4. Removed the server")
        output.print()

    except Exception as e:
        output.error(f"Demo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        output.warning("\n⚠️  Demo interrupted")
    except Exception as e:
        output.error(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
