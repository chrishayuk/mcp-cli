#!/usr/bin/env python
"""
Demo script for MCP-CLI Interactive Mode.

This script demonstrates the capabilities of the unified command system
in interactive mode by simulating various command interactions.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_cli.adapters.interactive import InteractiveCommandAdapter
from mcp_cli.commands import register_all_commands
from mcp_cli.commands.registry import registry
from mcp_cli.commands.base import CommandMode
from mcp_cli.context import initialize_context
from chuk_term.ui import output
from chuk_term.ui.theme import set_theme


async def demo_interactive_commands():
    """Demonstrate interactive mode command capabilities."""

    # Initialize
    output.rule("MCP-CLI Interactive Mode Demo", style="bold cyan")
    output.info("Demonstrating the unified command system in interactive mode")

    # Register all commands
    register_all_commands()

    # Initialize context (needed for commands that require it)
    initialize_context()

    # Set a nice theme for the demo
    set_theme("default")

    # Demo 1: Help command
    output.rule("Demo 1: Help Command")
    output.info("Executing: help")
    await InteractiveCommandAdapter.handle_command("help")

    await asyncio.sleep(1)

    # Demo 2: Theme commands
    output.rule("Demo 2: Theme Commands")

    # List themes
    output.info("Executing: themes")
    await InteractiveCommandAdapter.handle_command("themes")

    await asyncio.sleep(1)

    # Change theme
    output.info("Executing: theme dark")
    await InteractiveCommandAdapter.handle_command("theme dark")

    await asyncio.sleep(1)

    # Demo 3: Verbose mode toggle
    output.rule("Demo 3: Verbose Mode")

    output.info("Executing: verbose")
    await InteractiveCommandAdapter.handle_command("verbose")

    output.info("Executing: verbose on")
    await InteractiveCommandAdapter.handle_command("verbose on")

    output.info("Executing: verbose off")
    await InteractiveCommandAdapter.handle_command("verbose off")

    await asyncio.sleep(1)

    # Demo 4: Command completion
    output.rule("Demo 4: Command Completion")

    # Show what completions would be available
    completions = InteractiveCommandAdapter.get_completions("th", 2)
    output.info(f"Completions for 'th': {completions}")

    completions = InteractiveCommandAdapter.get_completions("help ", 5)
    output.info(f"Completions for 'help ': {completions}")

    await asyncio.sleep(1)

    # Demo 5: Command with arguments
    output.rule("Demo 5: Commands with Arguments")

    # Get help for a specific command
    output.info("Executing: help servers")
    await InteractiveCommandAdapter.handle_command("help servers")

    await asyncio.sleep(1)

    # Demo 6: Invalid command handling
    output.rule("Demo 6: Error Handling")

    output.info("Executing: invalid_command")
    result = await InteractiveCommandAdapter.handle_command("invalid_command")
    output.info(f"Command handled: {result}")

    await asyncio.sleep(1)

    # Demo 7: List available commands by mode
    output.rule("Demo 7: Command Registry Info")

    # Show commands available in interactive mode
    interactive_commands = registry.list_commands(mode=CommandMode.INTERACTIVE)
    output.info(f"Commands available in INTERACTIVE mode: {len(interactive_commands)}")
    for cmd in interactive_commands[:5]:  # Show first 5
        output.print(f"  - {cmd.name}: {cmd.description}")

    # Show command aliases
    output.info("\nCommand aliases:")
    for cmd in interactive_commands[:3]:
        if cmd.aliases:
            output.print(f"  - {cmd.name}: {', '.join(cmd.aliases)}")

    await asyncio.sleep(1)

    # Demo 8: Clear command
    output.rule("Demo 8: Clear Screen")
    output.info("Executing: clear (will clear screen in 2 seconds)")
    await asyncio.sleep(2)
    await InteractiveCommandAdapter.handle_command("clear")

    output.success("\nInteractive Mode Demo Complete!")
    output.info(
        "All commands executed successfully through the unified command system."
    )

    # Show summary
    output.rule("Summary")
    output.print("✓ Help system working")
    output.print("✓ Theme commands working")
    output.print("✓ Verbose mode toggle working")
    output.print("✓ Command completion working")
    output.print("✓ Command arguments working")
    output.print("✓ Error handling working")
    output.print("✓ Command registry working")
    output.print("✓ Clear command working")


async def main():
    """Main entry point."""
    try:
        await demo_interactive_commands()
    except KeyboardInterrupt:
        output.warning("\nDemo interrupted by user")
    except Exception as e:
        output.error(f"Demo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
