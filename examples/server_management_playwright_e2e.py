#!/usr/bin/env python
"""
End-to-end example using Playwright MCP server.

This script demonstrates:
1. Adding the Playwright MCP server at runtime
2. Listing servers to confirm addition
3. Showing server details
4. Removing the server

Playwright MCP server provides browser automation tools for web testing and scraping.
"""

import asyncio

# Initialize context before imports that need it
from mcp_cli.context import initialize_context

initialize_context()

from mcp_cli.commands.actions.servers import servers_action_async  # noqa: E402
from chuk_term.ui import output  # noqa: E402


async def main():
    """Run the Playwright server management demo."""

    output.rule("[bold magenta]🎭 Playwright MCP Server Demo[/bold magenta]")
    output.print()
    output.info("This demo shows how to add and manage the Playwright MCP server")
    output.info("Playwright provides browser automation capabilities via MCP tools")
    output.print()

    # Step 1: Show initial servers
    output.rule("[bold cyan]Step 1: Current Servers[/bold cyan]")
    await servers_action_async(args=["list"])
    output.print()

    # Step 2: Add Playwright server using npx
    output.rule("[bold cyan]Step 2: Adding Playwright MCP Server[/bold cyan]")
    output.info("Adding Playwright server with command: npx @playwright/mcp@latest")

    # Using the enhanced syntax with -- separator
    await servers_action_async(
        args=["add", "playwright", "stdio", "npx", "@playwright/mcp@latest"]
    )
    output.print()

    # Step 3: List servers to confirm addition
    output.rule("[bold cyan]Step 3: Servers After Adding Playwright[/bold cyan]")
    await servers_action_async(args=["list"])
    output.print()

    # Step 4: Show Playwright server details
    output.rule("[bold cyan]Step 4: Playwright Server Details[/bold cyan]")
    await servers_action_async(args=["playwright"])
    output.print()

    # Step 5: Demonstrate alternative add syntax with environment variables
    output.rule(
        "[bold cyan]Step 5: Adding Server with Environment Variables[/bold cyan]"
    )
    output.info("Example: Adding a server that needs API keys")

    await servers_action_async(
        args=[
            "add",
            "example-api",
            "--env",
            "API_KEY=demo-key-12345",
            "--env",
            "API_SECRET=demo-secret",
            "--",
            "stdio",
            "echo",
            "simulated-api-server",
        ]
    )
    output.print()

    # Step 6: Demonstrate HTTP/SSE server addition
    output.rule("[bold cyan]Step 6: Adding HTTP/SSE Servers[/bold cyan]")
    output.info("Example: Adding an HTTP server with headers")

    await servers_action_async(
        args=[
            "add",
            "example-http",
            "--transport",
            "http",
            "--header",
            "Authorization: Bearer demo-token",
            "--",
            "https://api.example.com/mcp",
        ]
    )
    output.print()

    # Step 7: List all servers including new ones
    output.rule("[bold cyan]Step 7: All Configured Servers[/bold cyan]")
    await servers_action_async(args=["list"])
    output.print()

    # Step 8: Clean up - remove demo servers
    output.rule("[bold cyan]Step 8: Cleanup - Removing Demo Servers[/bold cyan]")

    for server_name in ["playwright", "example-api", "example-http"]:
        output.info(f"Removing {server_name}...")
        await servers_action_async(args=["remove", server_name])

    output.print()

    # Step 9: Final server list
    output.rule("[bold cyan]Step 9: Final Server List[/bold cyan]")
    await servers_action_async(args=["list"])
    output.print()

    # Summary
    output.rule("[bold green]✅ Demo Complete![/bold green]")
    output.print()
    output.success("Successfully demonstrated Playwright server management:")
    output.info("  • Added Playwright MCP server using npx")
    output.info("  • Added server with environment variables")
    output.info("  • Added HTTP server with authentication headers")
    output.info("  • Listed and inspected servers")
    output.info("  • Removed all demo servers")
    output.print()

    output.tip("💡 Real-world usage examples:")
    output.print()

    output.info("1. Playwright for browser automation:")
    output.print("   /server add playwright stdio npx @playwright/mcp@latest")
    output.print()

    output.info("2. Airtable with API key:")
    output.print(
        "   /server add airtable --env AIRTABLE_API_KEY=your-key -- stdio npx -y airtable-mcp-server"
    )
    output.print()

    output.info("3. Linear SSE server:")
    output.print("   /server add linear --transport sse -- https://mcp.linear.app/sse")
    output.print()

    output.info("4. Private HTTP API with Bearer token:")
    output.print(
        '   /server add api --transport http --header "Authorization: Bearer token" -- https://api.company.com/mcp'
    )
    output.print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        output.warning("\n⚠️  Demo interrupted by user")
    except Exception as e:
        output.error(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
