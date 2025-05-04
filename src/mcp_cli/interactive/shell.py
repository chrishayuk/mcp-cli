# src/mcp_cli/interactive/shell.py
"""Interactive shell implementation for MCP CLI."""
from __future__ import annotations
import asyncio
import logging
import shlex
from typing import Any, Dict, List, Optional

from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# mcp cli
from mcp_cli.tools.manager import ToolManager
from mcp_cli.ui.ui_helpers import display_welcome_banner

# commands
from mcp_cli.interactive.commands import register_all_commands
from mcp_cli.interactive.registry import InteractiveCommandRegistry

# logger
logger = logging.getLogger(__name__)


async def interactive_mode(
    stream_manager: Any = None,
    tool_manager: Optional[ToolManager] = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    server_names: Optional[Dict[int, str]] = None,
    **kwargs
) -> bool:
    """
    Launch the interactive mode CLI.

    Parameters
    ----------
    stream_manager
        StreamManager instance with connections to all servers.
    tool_manager
        Optional ToolManager instance.
    provider / model
        LLM configuration for system display.
    server_names
        Optional mapping of server indices to names.
    **kwargs
        Additional parameters passed from run_command.
    """
    console = Console()

    # Register all commands
    register_all_commands()

    # Display welcome banner
    display_welcome_banner({"provider": provider, "model": model})

    # Intro panel
    print(Panel(
        Markdown(
            "# Interactive Mode\n\n"
            "Type commands to interact with the system.\n"
            "Type 'help' to see available commands.\n"
            "Type 'exit' or 'quit' to exit."
        ),
        title="MCP Interactive Mode",
        style="bold cyan"
    ))

    # Initial help
    help_cmd = InteractiveCommandRegistry.get_command("help")
    if help_cmd:
        await help_cmd.execute([], tool_manager, server_names=server_names)

    # Main loop
    while True:
        try:
            # Read a line
            raw = await asyncio.to_thread(input, "> ")
            line = raw.strip()

            # Skip empty
            if not line:
                continue

            # If user just types "/", re-show the commands list
            if line == "/":
                if help_cmd:
                    await help_cmd.execute([], tool_manager, server_names=server_names)
                continue

            # Parse
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = line.split()

            cmd_name = parts[0].lstrip("/").lower()
            args = parts[1:]

            # Lookup
            cmd = InteractiveCommandRegistry.get_command(cmd_name)
            if cmd:
                result = await cmd.execute(args, tool_manager, server_names=server_names, **kwargs)
                if result is True:
                    return True
            else:
                print(f"[red]Unknown command: {cmd_name}[/red]")
                print("[dim]Type 'help' to see available commands.[/dim]")

        except KeyboardInterrupt:
            print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except EOFError:
            print("\n[yellow]EOF detected. Exiting.[/yellow]")
            return True
        except Exception as e:
            logger.exception("Error in interactive mode")
            print(f"[red]Error: {e}[/red]")

    return True
