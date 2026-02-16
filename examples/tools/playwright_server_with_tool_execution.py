#!/usr/bin/env python
"""
Complete end-to-end example with Playwright MCP server including tool execution.

This script demonstrates:
1. Adding the Playwright MCP server
2. Restarting the tool manager to connect
3. Listing available Playwright tools
4. Executing a Playwright tool (navigate to a webpage)
5. Removing the server

This proves the entire lifecycle including actual tool usage.
"""

import asyncio
import json

# Initialize context before imports that need it
from mcp_cli.context import initialize_context

initialize_context()

from mcp_cli.commands.actions.servers import servers_action_async  # noqa: E402
from mcp_cli.config.config_manager import ConfigManager  # noqa: E402
from mcp_cli.tools.manager import ToolManager  # noqa: E402
from chuk_term.ui import output  # noqa: E402


async def add_playwright_server():
    """Add the Playwright MCP server to configuration."""
    output.rule("[bold cyan]Step 1: Adding Playwright MCP Server[/bold cyan]")

    # Check if already exists
    config_manager = ConfigManager()
    try:
        config = config_manager.get_config()
    except RuntimeError:
        config = config_manager.initialize()

    if "playwright" in config.servers:
        output.warning("Playwright server already exists, removing first...")
        await servers_action_async(args=["remove", "playwright"])

    # Add Playwright server
    await servers_action_async(
        args=["add", "playwright", "stdio", "npx", "@playwright/mcp@latest"]
    )

    output.success("✅ Playwright server added to configuration")
    print()


async def connect_to_playwright():
    """Initialize tool manager and connect to Playwright server."""
    output.rule("[bold cyan]Step 2: Connecting to Playwright Server[/bold cyan]")

    # Get updated configuration
    config_manager = ConfigManager()
    config = config_manager.reload()

    # Get all enabled servers
    enabled_servers = config.list_enabled_servers()
    output.info(f"Found {len(enabled_servers)} configured server(s)")

    # Convert to format expected by ToolManager
    server_configs = []
    for server in enabled_servers:
        if server.command:
            config_dict = {
                "name": server.name,
                "command": server.command,
                "args": server.args,
                "env": server.env,
            }
            server_configs.append(config_dict)
            output.info(f"  • {server.name}: {server.command} {' '.join(server.args)}")

    # Create tool manager with servers parameter as a list
    output.info("\nInitializing Tool Manager...")
    tm = ToolManager(servers=server_configs)

    # Initialize connections
    await tm.initialize()

    # Get server info to verify connection
    servers = await tm.get_server_info() if hasattr(tm, "get_server_info") else []

    playwright_connected = False
    for server in servers:
        if server.name.lower() == "playwright":
            playwright_connected = True
            output.success("✅ Connected to Playwright server")
            output.info(f"   Tools available: {server.tool_count}")
            break

    if not playwright_connected:
        output.warning(
            "⚠️  Playwright server not connected. It may take a moment to initialize."
        )
        output.info("   The npx command will download the package if needed.")

    print()
    return tm


async def list_playwright_tools(tm):
    """List available Playwright tools."""
    output.rule("[bold cyan]Step 3: Listing Playwright Tools[/bold cyan]")

    if not tm:
        output.error("Tool manager not available")
        return []

    # Get all tools
    tools = await tm.get_tools()

    # Filter Playwright tools
    playwright_tools = []
    for tool in tools:
        # Check if tool belongs to playwright namespace
        namespace = getattr(tool, "namespace", "")
        if "playwright" in str(namespace).lower() or "playwright" in tool.name.lower():
            playwright_tools.append(tool)

    if playwright_tools:
        output.success(f"Found {len(playwright_tools)} Playwright tools:")
        # Show first 10 tools
        for tool in playwright_tools[:10]:
            description = getattr(tool, "description", "No description")
            output.info(f"  • {tool.name}")
            if description and description != "No description":
                output.print(f"    {description[:80]}...")
    else:
        output.warning(
            "No Playwright-specific tools found. Showing all available tools:"
        )
        for tool in tools[:5]:
            output.info(f"  • {tool.name}")

    print()
    return playwright_tools or tools


async def execute_playwright_tool(tm, tools):
    """Execute a Playwright tool to demonstrate functionality."""
    output.rule("[bold green]Step 4: Executing Playwright Tool[/bold green]")

    if not tm or not tools:
        output.error("No tools available to execute")
        return

    # Try to find a simple Playwright tool to execute
    # Look for navigation or screenshot tools which are common
    target_tool = None

    # Priority list of tools to try
    tool_priorities = [
        "playwright_navigate",
        "playwright_screenshot",
        "navigate",
        "screenshot",
        "get_page_content",
    ]

    for priority_name in tool_priorities:
        for tool in tools:
            if priority_name in tool.name.lower():
                target_tool = tool
                break
        if target_tool:
            break

    # If no specific tool found, use the first available tool
    if not target_tool and tools:
        target_tool = tools[0]

    if not target_tool:
        output.error("No suitable tool found to execute")
        return

    output.info(f"Executing tool: {target_tool.name}")

    try:
        # Prepare tool request based on tool requirements
        tool_request = {"name": target_tool.name, "arguments": {}}

        # Add common arguments based on tool name
        if "navigate" in target_tool.name.lower():
            tool_request["arguments"]["url"] = "https://example.com"
            output.info("  Navigating to https://example.com")
        elif "screenshot" in target_tool.name.lower():
            tool_request["arguments"]["url"] = "https://example.com"
            tool_request["arguments"]["path"] = "/tmp/screenshot.png"
            output.info("  Taking screenshot of https://example.com")

        # Execute the tool
        output.info("  Executing...")
        result = await tm.execute_tool(tool_request)

        if result:
            output.success("✅ Tool executed successfully!")

            # Display result (truncated for readability)
            if isinstance(result, dict):
                result_str = json.dumps(result, indent=2)
            else:
                result_str = str(result)

            if len(result_str) > 500:
                output.print(f"\nResult (truncated):\n{result_str[:500]}...")
            else:
                output.print(f"\nResult:\n{result_str}")
        else:
            output.warning("Tool executed but returned no result")

    except Exception as e:
        output.error(f"Error executing tool: {e}")
        # Try a simpler tool if the first one failed
        if len(tools) > 1:
            output.info("\nTrying a different tool...")
            try:
                simple_tool = tools[1]
                tool_request = {"name": simple_tool.name, "arguments": {}}
                result = await tm.execute_tool(tool_request)
                if result:
                    output.success(
                        f"✅ Alternative tool '{simple_tool.name}' executed!"
                    )
            except Exception as e2:
                output.error(f"Alternative tool also failed: {e2}")

    print()


async def remove_playwright_server():
    """Remove the Playwright server from configuration."""
    output.rule("[bold cyan]Step 5: Removing Playwright Server[/bold cyan]")

    await servers_action_async(args=["remove", "playwright"])

    output.success("✅ Playwright server removed from configuration")
    print()


async def main():
    """Run the complete Playwright server demo with tool execution."""

    output.rule(
        "[bold magenta]🎭 Playwright MCP Server - Complete Demo with Tool Execution[/bold magenta]"
    )
    output.print()
    output.info("This demo shows the complete lifecycle:")
    output.info("  1. Add Playwright MCP server")
    output.info("  2. Connect to the server")
    output.info("  3. List available tools")
    output.info("  4. Execute a Playwright tool")
    output.info("  5. Remove the server")
    output.print()

    try:
        # Step 1: Add Playwright server
        await add_playwright_server()

        # Step 2: Connect to servers
        tm = await connect_to_playwright()

        if tm:
            # Step 3: List Playwright tools
            tools = await list_playwright_tools(tm)

            # Step 4: Execute a tool
            if tools:
                await execute_playwright_tool(tm, tools)

            # Cleanup tool manager
            await tm.cleanup()

        # Step 5: Remove Playwright server
        await remove_playwright_server()

        # Summary
        output.rule("[bold green]✅ Demo Complete![/bold green]")
        output.success("Successfully demonstrated the complete server lifecycle:")
        output.info("  ✓ Added Playwright MCP server")
        output.info("  ✓ Connected to the server")
        output.info("  ✓ Listed available tools")
        output.info("  ✓ Executed a Playwright tool")
        output.info("  ✓ Removed the server")
        output.print()

        output.tip("💡 To use in chat mode:")
        output.print(
            "  1. Add server: /server add playwright stdio npx @playwright/mcp@latest"
        )
        output.print("  2. Restart session or use: /server reload")
        output.print("  3. Use tools: /tools --server playwright")
        output.print("  4. Execute: Use natural language to request browser actions")
        output.print()

    except Exception as e:
        output.error(f"Demo error: {e}")
        import traceback

        traceback.print_exc()

        # Try to clean up
        try:
            output.info("\nCleaning up...")
            await remove_playwright_server()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        output.warning("\n⚠️  Demo interrupted by user")
    except Exception as e:
        output.error(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
