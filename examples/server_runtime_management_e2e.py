#!/usr/bin/env python
"""
Complete end-to-end example of runtime server management.

This demonstrates:
1. Adding servers to user configuration (~/.mcp-cli)
2. Listing servers from both project config and user preferences
3. Managing server state (enable/disable)
4. Removing servers from user configuration

The user configuration persists across sessions, while project config
remains in server_config.json.
"""

import asyncio
from mcp_cli.context import initialize_context
from mcp_cli.commands.actions.servers import servers_action_async  # noqa: E402
from mcp_cli.utils.preferences import get_preference_manager  # noqa: E402
from chuk_term.ui import output  # noqa: E402


async def main():
    """Run the complete server runtime management demo."""

    output.rule("[bold magenta]🚀 Runtime Server Management Demo[/bold magenta]")
    output.print()
    output.info("This demo shows how servers are managed at runtime:")
    output.info("  • Project servers: Stored in server_config.json")
    output.info("  • User servers: Stored in ~/.mcp-cli/preferences.json")
    output.print()

    # Initialize context
    initialize_context()
    pref_manager = get_preference_manager()

    # Step 1: Show initial state
    output.rule("[bold cyan]Step 1: Initial Server Configuration[/bold cyan]")
    await servers_action_async(args=["list"])
    output.print()

    # Step 2: Add Playwright server
    output.rule("[bold cyan]Step 2: Adding Playwright Server (STDIO)[/bold cyan]")
    output.info("Adding npx @playwright/mcp@latest as a user server...")
    await servers_action_async(
        args=["add", "playwright", "stdio", "npx", "@playwright/mcp@latest"]
    )
    output.print()

    # Step 3: Add HTTP API server with authentication
    output.rule("[bold cyan]Step 3: Adding HTTP API Server with Auth[/bold cyan]")
    output.info("Adding an HTTP server with Bearer token authentication...")
    await servers_action_async(
        args=[
            "add",
            "github-api",
            "--transport",
            "http",
            "--header",
            "Authorization: Bearer ghp_demo_token_12345",
            "--env",
            "GITHUB_USER=demo",
            "--",
            "https://api.github.com/mcp",
        ]
    )
    output.print()

    # Step 4: Add SSE server
    output.rule("[bold cyan]Step 4: Adding SSE Server[/bold cyan]")
    output.info("Adding a Server-Sent Events server...")
    await servers_action_async(
        args=["add", "linear", "--transport", "sse", "--", "https://mcp.linear.app/sse"]
    )
    output.print()

    # Step 5: List all servers
    output.rule("[bold cyan]Step 5: All Configured Servers[/bold cyan]")
    await servers_action_async(args=["list"])
    output.print()

    # Step 6: Show server details
    output.rule("[bold cyan]Step 6: Server Details[/bold cyan]")
    output.info("Showing details for github-api server...")
    await servers_action_async(args=["github-api"])
    output.print()

    # Step 7: Disable a server
    output.rule("[bold cyan]Step 7: Disabling a Server[/bold cyan]")
    output.info("Disabling the linear server...")
    await servers_action_async(args=["disable", "linear"])
    output.print()

    # Step 8: List servers (disabled hidden by default)
    output.rule("[bold cyan]Step 8: Active Servers Only[/bold cyan]")
    await servers_action_async(args=["list"])
    output.print()

    # Step 9: List all servers including disabled
    output.rule("[bold cyan]Step 9: All Servers (Including Disabled)[/bold cyan]")
    await servers_action_async(args=["list", "all"])
    output.print()

    # Step 10: Re-enable server
    output.rule("[bold cyan]Step 10: Re-enabling Server[/bold cyan]")
    output.info("Re-enabling the linear server...")
    await servers_action_async(args=["enable", "linear"])
    output.print()

    # Step 11: Check persistence
    output.rule("[bold cyan]Step 11: Verifying Persistence[/bold cyan]")
    runtime_servers = pref_manager.get_runtime_servers()
    output.success(
        f"✅ {len(runtime_servers)} user servers stored in ~/.mcp-cli/preferences.json:"
    )
    for name, config in runtime_servers.items():
        output.info(f"  • {name}: {config.get('transport', 'stdio').upper()}")
    output.print()

    # Step 12: Clean up
    output.rule("[bold cyan]Step 12: Cleanup[/bold cyan]")
    output.info("Removing user-added servers...")
    for server in ["playwright", "github-api", "linear"]:
        await servers_action_async(args=["remove", server])
    output.print()

    # Final state
    output.rule("[bold cyan]Final State[/bold cyan]")
    await servers_action_async(args=["list"])
    output.print()

    # Summary
    output.rule("[bold green]✅ Demo Complete![/bold green]")
    output.success("Key takeaways:")
    output.info("  1. User servers persist in ~/.mcp-cli/preferences.json")
    output.info("  2. Project servers stay in server_config.json")
    output.info("  3. Servers can be enabled/disabled without removal")
    output.info("  4. Supports STDIO, HTTP, and SSE transports")
    output.info("  5. Environment variables and headers are supported")
    output.print()

    output.tip("💡 Real-world examples:")
    output.print()
    output.info("Playwright (browser automation):")
    output.print("  /server add playwright stdio npx @playwright/mcp@latest")
    output.print()
    output.info("GitHub with token:")
    output.print(
        "  /server add github --env GITHUB_TOKEN=token -- stdio npx github-mcp"
    )
    output.print()
    output.info("Custom API with Bearer auth:")
    output.print(
        "  /server add api --transport http --header 'Authorization: Bearer token' -- https://api.example.com/mcp"
    )
    output.print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        output.warning("\n⚠️  Demo interrupted")
    except Exception as e:
        output.error(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
