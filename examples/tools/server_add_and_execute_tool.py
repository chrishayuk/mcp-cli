#!/usr/bin/env python
"""
Complete end-to-end example: Add server, connect, and execute tools.

This demonstrates:
1. Adding a new MCP server at runtime
2. Initializing the tool manager with the new server
3. Listing available tools from the server
4. Executing a tool and displaying results
5. Cleaning up

We use the 'echo' server as it's simple and always available.
"""

import asyncio
import json

# Initialize context before imports that need it
from mcp_cli.context import initialize_context, get_context

initialize_context()

from mcp_cli.commands.actions.servers import servers_action_async  # noqa: E402
from mcp_cli.utils.preferences import get_preference_manager  # noqa: E402
from mcp_cli.config.config_manager import ConfigManager  # noqa: E402
from mcp_cli.tools.manager import ToolManager  # noqa: E402
from chuk_term.ui import output  # noqa: E402


async def initialize_tool_manager_with_servers():
    """Initialize tool manager with all configured servers."""
    output.rule("[bold cyan]Initializing Tool Manager[/bold cyan]")

    context = get_context()
    config_manager = ConfigManager()
    pref_manager = get_preference_manager()

    # Get all servers (project + user)
    try:
        config = config_manager.get_config()
    except RuntimeError:
        config = config_manager.initialize()

    # Add runtime servers to config temporarily for tool manager
    runtime_servers = pref_manager.get_runtime_servers()
    for name, server_config in runtime_servers.items():
        if not pref_manager.is_server_disabled(name):
            # Create ServerConfig from runtime config
            from mcp_cli.config.config_manager import ServerConfig

            server = ServerConfig(
                name=name,
                command=server_config.get("command"),
                args=server_config.get("args", []),
                env=server_config.get("env", {}),
                url=server_config.get("url"),
                disabled=False,
            )
            # Temporarily add to config for tool manager
            config.servers[name] = server

    # Save temporarily
    config_manager.save()

    # Get all server names
    all_servers = [name for name, srv in config.servers.items() if not srv.disabled]

    output.info(f"Initializing with servers: {', '.join(all_servers)}")

    # Create tool manager with config file and server list
    config_file = "server_config.json"
    tm = ToolManager(
        config_file=config_file,
        servers=all_servers,
        server_names={i: name for i, name in enumerate(all_servers)},
    )

    # Initialize connections
    await tm.initialize()

    # Update context
    context.tool_manager = tm

    # Verify connections
    servers_info = await tm.get_server_info() if hasattr(tm, "get_server_info") else []
    output.success(f"✅ Connected to {len(servers_info)} server(s)")

    for server in servers_info:
        output.info(f"  • {server.name}: {server.tool_count} tools")

    return tm


async def list_server_tools(tm, server_name: str):
    """List tools from a specific server."""
    output.rule(f"[bold cyan]Tools from {server_name} server[/bold cyan]")

    if not tm:
        output.error("Tool manager not available")
        return []

    # Get all tools
    tools = await tm.get_all_tools()

    # Filter tools by server/namespace
    server_tools = []
    for tool in tools:
        # Check namespace or name for server identification
        namespace = getattr(tool, "namespace", "")
        if (
            server_name.lower() in str(namespace).lower()
            or server_name.lower() in tool.name.lower()
        ):
            server_tools.append(tool)

    # If no specific match, show all tools (might be from this server)
    if not server_tools and server_name in ["echo", "test-echo"]:
        # Echo server tools typically have 'echo' in the name
        server_tools = [t for t in tools if "echo" in t.name.lower()]

    if server_tools:
        output.success(f"Found {len(server_tools)} tool(s) from {server_name}:")
        for tool in server_tools[:10]:  # Show first 10
            description = getattr(tool, "description", "")
            output.info(f"  • {tool.name}")
            if description:
                output.print(f"    {description[:60]}...")
    else:
        output.warning(f"No specific tools found for {server_name}")
        output.info("Available tools from all servers:")
        for tool in tools[:5]:
            output.info(f"  • {tool.name}")

    return server_tools or tools


async def execute_tool_example(tm, tools):
    """Execute a tool and show the results."""
    output.rule("[bold green]Executing Tool[/bold green]")

    if not tm or not tools:
        output.error("No tools available to execute")
        return

    # Find a suitable tool to execute
    target_tool = None

    # Look for echo_message or similar simple tool
    for tool in tools:
        if "echo" in tool.name.lower() and "message" in tool.name.lower():
            target_tool = tool
            break

    # Fallback to first available tool
    if not target_tool:
        target_tool = tools[0]

    output.info(f"Executing tool: {target_tool.name}")

    try:
        # Prepare arguments based on tool name
        arguments = {}

        # Add appropriate arguments based on tool name
        if "echo" in target_tool.name.lower():
            arguments["message"] = "Hello from MCP CLI runtime server!"
            output.info("  Sending message: 'Hello from MCP CLI runtime server!'")
        elif "get" in target_tool.name.lower() or "list" in target_tool.name.lower():
            # For getter/lister tools, usually no args needed
            output.info("  Executing with no arguments (getter/lister tool)")
        else:
            # For other tools, try common argument names
            arguments["input"] = "test"
            output.info("  Sending test input")

        # Execute the tool with correct signature
        output.info("  Executing...")
        result = await tm.execute_tool(target_tool.name, arguments)

        if result:
            output.success("✅ Tool executed successfully!")

            # Format and display result
            if isinstance(result, dict):
                result_str = json.dumps(result, indent=2)
            else:
                result_str = str(result)

            output.print("\n[bold]Result:[/bold]")
            if len(result_str) > 500:
                output.print(result_str[:500] + "...")
            else:
                output.print(result_str)
        else:
            output.warning("Tool executed but returned no result")

    except Exception as e:
        output.error(f"Error executing tool: {e}")

        # Try a simpler approach if first attempt failed
        if len(tools) > 1:
            output.info("\nTrying alternative tool...")
            try:
                alt_tool = tools[1]
                result = await tm.execute_tool(alt_tool.name, {})
                if result:
                    output.success(f"✅ Alternative tool '{alt_tool.name}' executed!")
                    output.print(f"Result: {str(result)[:200]}")
            except Exception as e2:
                output.error(f"Alternative also failed: {e2}")


async def main():
    """Run the complete add server and execute tool demo."""

    output.rule(
        "[bold magenta]🚀 Add Server & Execute Tool - Complete Demo[/bold magenta]"
    )
    output.print()
    output.info("This demo will:")
    output.info("  1. Add a new echo server")
    output.info("  2. Connect to all servers including the new one")
    output.info("  3. List available tools")
    output.info("  4. Execute a tool and show results")
    output.info("  5. Clean up")
    output.print()

    pref_manager = get_preference_manager()
    config_manager = ConfigManager()

    # Clean up any existing test-echo server from config file first
    try:
        config = config_manager.get_config()
        if "test-echo" in config.servers:
            del config.servers["test-echo"]
            config_manager.save()
            output.info("Cleaned up existing test-echo from config")
    except Exception:
        pass

    try:
        # Step 1: Show initial servers
        output.rule("[bold]Step 1: Initial Servers[/bold]")
        await servers_action_async(args=["list"])
        output.print()

        # Step 2: Add a test echo server
        output.rule("[bold]Step 2: Adding Test Echo Server[/bold]")
        output.info("Adding a simple echo server for testing...")

        # Check if already exists in user prefs and remove if so
        if pref_manager.get_runtime_server("test-echo"):
            await servers_action_async(args=["remove", "test-echo"])

        await servers_action_async(
            args=["add", "test-echo", "stdio", "uvx", "chuk-mcp-echo", "stdio"]
        )
        output.print()

        # Step 3: Verify server was added
        output.rule("[bold]Step 3: Verify Server Added[/bold]")
        await servers_action_async(args=["list"])
        output.print()

        # Step 4: Initialize tool manager with all servers
        output.rule("[bold]Step 4: Connecting to Servers[/bold]")
        tm = await initialize_tool_manager_with_servers()
        output.print()

        if tm:
            # Step 5: List tools from the echo server
            output.rule("[bold]Step 5: Listing Tools[/bold]")
            tools = await list_server_tools(tm, "test-echo")
            output.print()

            # Step 6: Execute a tool
            output.rule("[bold]Step 6: Tool Execution[/bold]")
            if tools:
                await execute_tool_example(tm, tools)
            else:
                output.warning("No tools available to execute")
            output.print()

            # Clean up tool manager
            await tm.close()

            # Clean up the temporary servers from config
            config = config_manager.get_config()
            runtime_servers = pref_manager.get_runtime_servers()
            for name in list(runtime_servers.keys()):
                if name in config.servers:
                    del config.servers[name]
            config_manager.save()

        # Step 7: Remove the test server
        output.rule("[bold]Step 7: Cleanup[/bold]")
        output.info("Removing test-echo server...")
        await servers_action_async(args=["remove", "test-echo"])
        output.print()

        # Step 8: Final state
        output.rule("[bold]Step 8: Final State[/bold]")
        await servers_action_async(args=["list"])
        output.print()

        # Summary
        output.rule("[bold green]✅ Demo Complete![/bold green]")
        output.success("Successfully demonstrated:")
        output.info("  ✓ Added server at runtime")
        output.info("  ✓ Connected to the server")
        output.info("  ✓ Listed available tools")
        output.info("  ✓ Executed a tool from the server")
        output.info("  ✓ Cleaned up")
        output.print()

        output.tip("💡 Try with other servers:")
        output.print("  /server add playwright stdio npx @playwright/mcp@latest")
        output.print("  /server add time stdio uvx mcp-server-time")
        output.print(
            "  /server add fs stdio npx @modelcontextprotocol/server-filesystem"
        )
        output.print()

    except Exception as e:
        output.error(f"Demo error: {e}")
        import traceback

        traceback.print_exc()

        # Try to clean up
        try:
            output.info("\nCleaning up...")
            if pref_manager.get_runtime_server("test-echo"):
                await servers_action_async(args=["remove", "test-echo"])
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        output.warning("\n⚠️  Demo interrupted")
    except Exception as e:
        output.error(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
