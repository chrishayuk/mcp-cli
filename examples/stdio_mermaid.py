#!/usr/bin/env python3
"""
Test Mermaid MCP Server STDIO Connection

This script tests the basic STDIO connection to the mermaid server
to isolate issues from the main mcp-cli application.

Similar to the chuk-tool-processor sqlite example but for mermaid.
"""

import asyncio
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def test_mermaid_stdio():
    """Test STDIO connection to mermaid server"""

    console.print(Panel.fit(
        "[bold cyan]Mermaid MCP Server STDIO Test[/bold cyan]\n\n"
        "Testing basic STDIO connection and tool retrieval",
        border_style="cyan"
    ))

    # Step 1: Load config
    console.print("\n[1/4] [bold]Loading server configuration...[/bold]")
    config_path = Path(__file__).parent.parent / "server_config.json"

    if not config_path.exists():
        console.print(f"[red]❌ Config not found: {config_path}[/red]")
        return False

    with open(config_path) as f:
        config = json.load(f)

    mermaid_config = config.get("mcpServers", {}).get("mermaid")
    if not mermaid_config:
        console.print("[red]❌ Mermaid server not found in config[/red]")
        return False

    console.print(f"    [green]✓[/green] Found mermaid config:")
    console.print(f"      Command: {mermaid_config.get('command')}")
    console.print(f"      Args: {mermaid_config.get('args')}")
    console.print(f"      Env: {mermaid_config.get('env', {})}")

    # Step 2: Initialize ToolManager
    console.print("\n[2/4] [bold]Initializing ToolManager...[/bold]")

    from mcp_cli.tools.manager import ToolManager

    tm = ToolManager(
        config_file=str(config_path),
        servers=["mermaid"],
        server_names={0: "mermaid"},
        initialization_timeout=180.0,  # Increase timeout for debugging
    )

    console.print("    [cyan]Attempting to initialize with 180s timeout...[/cyan]")
    console.print("    [yellow]This may take a while on first run...[/yellow]")

    init_success = await tm.initialize(namespace="stdio")

    if not init_success:
        console.print("[red]❌ Failed to initialize ToolManager[/red]")
        console.print("[yellow]Check if the mermaid server starts correctly[/yellow]")
        await tm.close()
        return False

    console.print("    [green]✓[/green] ToolManager initialized successfully")

    # Step 3: Get tools
    console.print("\n[3/4] [bold]Retrieving available tools...[/bold]")

    try:
        tools = await tm.get_all_tools()
        console.print(f"    [green]✓[/green] Retrieved {len(tools)} tools")

        if tools:
            table = Table(title="Available Mermaid Tools", show_header=True)
            table.add_column("Namespace", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Description", style="yellow")

            for tool in tools:
                table.add_row(
                    tool.namespace or "N/A",
                    tool.name,
                    tool.description or "No description"
                )

            console.print(table)
        else:
            console.print("[yellow]⚠ No tools found[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Error getting tools: {e}[/red]")
        await tm.close()
        return False

    # Step 4: Test a simple tool call (if available)
    console.print("\n[4/4] [bold]Testing tool execution...[/bold]")

    if tools:
        # Try to find a simple tool to test
        test_tool = tools[0]  # Use first available tool
        console.print(f"    Testing: [cyan]{test_tool.namespace}.{test_tool.name}[/cyan]")

        try:
            # Try with minimal arguments
            result = await tm.run_tool(f"{test_tool.namespace}.{test_tool.name}", {})
            console.print(f"    [green]✓[/green] Tool executed successfully")
            console.print(f"    Result: {result}")
        except Exception as e:
            console.print(f"    [yellow]⚠[/yellow] Tool execution failed (expected if args required): {e}")

    # Cleanup
    await tm.close()

    console.print(Panel.fit(
        "[bold green]✅ SUCCESS![/bold green]\n\n"
        "STDIO connection to mermaid server is working correctly",
        border_style="green"
    ))

    return True


async def main():
    """Main entry point"""
    try:
        success = await test_mermaid_stdio()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
