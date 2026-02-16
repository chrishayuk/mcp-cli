#!/usr/bin/env python
"""
Demo: Clear command with welcome banner

This example demonstrates the enhanced clear command that displays
the welcome banner after clearing the screen, showing the current
provider, model, and tool count.
"""

import asyncio
from chuk_term.ui import output, display_chat_banner
from mcp_cli.commands.actions.clear import clear_action


async def main():
    """Demo the clear command with banner."""

    # Initial banner
    output.print("[bold cyan]MCP CLI Clear Command Demo[/bold cyan]\n")

    # Display initial state
    display_chat_banner(
        provider="ollama",
        model="gpt-oss",
        additional_info={"Tools": "12", "API Base": "http://localhost:11434"},
    )

    # Simulate some chat activity
    output.print("\n[dim]User:[/dim] What is the weather today?")
    output.print(
        "[dim]Assistant:[/dim] I don't have access to real-time weather data..."
    )
    output.print("\n[dim]User:[/dim] Can you help me with Python?")
    output.print("[dim]Assistant:[/dim] Of course! I'd be happy to help with Python...")

    # Show that screen has content
    output.print("\n[yellow]Screen has accumulated some content...[/yellow]")
    await asyncio.sleep(2)

    # Clear and redisplay banner
    output.print("\n[cyan]Executing /clear command...[/cyan]")
    await asyncio.sleep(1)

    clear_action(
        show_banner=True,
        provider="ollama",
        model="gpt-oss",
        additional_info={"Tools": "12", "API Base": "http://localhost:11434"},
    )

    output.print("\n[green]✓ Screen cleared and banner redisplayed![/green]")
    output.print("[dim]Ready for new conversation...[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
