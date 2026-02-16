#!/usr/bin/env python
"""
End-to-end example of MCP server runtime management.

This script demonstrates:
1. Listing current servers
2. Adding a new MCP server at runtime
3. Using the newly added server (calling its tools)
4. Disabling the server
5. Re-enabling the server
6. Removing the server
"""

import asyncio
from pathlib import Path

# Initialize context before imports that need it
from mcp_cli.context import initialize_context

initialize_context()

from mcp_cli.commands.actions.servers import servers_action_async  # noqa: E402
from mcp_cli.context import get_context  # noqa: E402
from chuk_term.ui import output  # noqa: E402


async def show_server_status(title: str):
    """Display current server status."""
    output.rule(f"[bold cyan]{title}[/bold cyan]")
    await servers_action_async(args=["list"])
    print()


async def test_server_tools(server_name: str):
    """Test tools from a specific server."""
    output.rule(f"[bold green]Testing tools from {server_name}[/bold green]")

    context = get_context()
    tm = context.tool_manager

    if not tm:
        output.error("Tool manager not available")
        return False

    try:
        # Get server info
        servers = await tm.get_server_info() if hasattr(tm, "get_server_info") else []

        # Find our server
        target_server = None
        for server in servers:
            if server.name.lower() == server_name.lower():
                target_server = server
                break

        if not target_server:
            output.warning(f"Server '{server_name}' not connected yet")
            output.info("Note: New servers require a session restart to connect")
            return False

        output.success(f"Found server: {target_server.name}")
        output.info(f"  Transport: {target_server.transport}")
        output.info(f"  Tools available: {target_server.tool_count}")

        # Get tools for this server
        if hasattr(tm, "get_tools_for_server"):
            tools = await tm.get_tools_for_server(server_name)
            if tools:
                output.info(
                    f"  First 3 tools: {', '.join([t.name for t in tools[:3]])}"
                )

        return True

    except Exception as e:
        output.error(f"Error testing server: {e}")
        return False


async def main():
    """Run the end-to-end server management demo."""

    output.rule("[bold magenta]🚀 MCP Server Management E2E Demo[/bold magenta]")
    output.print()

    # Step 1: Show initial servers
    await show_server_status("Step 1: Initial Server List")

    # Step 2: Add a new echo server
    output.rule("[bold cyan]Step 2: Adding a New Server[/bold cyan]")
    output.info("Adding 'demo-echo' server with stdio transport...")

    # We'll add the echo server which is simple and available
    await servers_action_async(
        args=["add", "demo-echo", "stdio", "uvx", "chuk-mcp-echo", "stdio"]
    )
    print()

    # Step 3: Show servers after adding
    await show_server_status("Step 3: Server List After Adding")

    # Step 4: Show details of the new server
    output.rule("[bold cyan]Step 4: Server Details[/bold cyan]")
    await servers_action_async(args=["demo-echo"])
    print()

    # Step 5: Try to use the server (note: requires restart to actually connect)
    await test_server_tools("demo-echo")
    print()

    # Step 6: Disable the server
    output.rule("[bold cyan]Step 6: Disabling Server[/bold cyan]")
    output.info("Disabling 'demo-echo' server...")
    await servers_action_async(args=["disable", "demo-echo"])
    print()

    # Step 7: Show servers with disabled server
    await show_server_status("Step 7: Server List with Disabled Server")

    # Step 8: Re-enable the server
    output.rule("[bold cyan]Step 8: Re-enabling Server[/bold cyan]")
    output.info("Re-enabling 'demo-echo' server...")
    await servers_action_async(args=["enable", "demo-echo"])
    print()

    # Step 9: Update server configuration
    output.rule("[bold cyan]Step 9: Updating Server Configuration[/bold cyan]")
    output.info("Adding environment variable to server...")
    await servers_action_async(args=["update", "demo-echo", "env", "DEBUG=true"])
    print()

    # Step 10: Remove the server
    output.rule("[bold cyan]Step 10: Removing Server[/bold cyan]")
    output.info("Removing 'demo-echo' server...")
    await servers_action_async(args=["remove", "demo-echo"])
    print()

    # Step 11: Final server list
    await show_server_status("Step 11: Final Server List")

    # Summary
    output.rule("[bold green]✅ Demo Complete![/bold green]")
    output.print()
    output.success("Successfully demonstrated server management lifecycle:")
    output.info("  • Listed existing servers")
    output.info("  • Added a new server (demo-echo)")
    output.info("  • Showed server details")
    output.info("  • Disabled and re-enabled the server")
    output.info("  • Updated server configuration")
    output.info("  • Removed the server")
    output.print()
    output.tip(
        "💡 Note: To actually connect to newly added servers and use their tools,"
    )
    output.tip("   you need to restart the chat session or use '/server reload'")
    output.print()

    # Show configuration file location
    config_path = Path("server_config.json").absolute()
    output.info(f"📁 Configuration saved to: {config_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        output.warning("\n⚠️  Demo interrupted by user")
    except Exception as e:
        output.error(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
