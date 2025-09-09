# src/mcp_cli/interactive/shell.py
"""Interactive shell implementation for MCP CLI with slash-menu autocompletion."""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List, Optional

from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

# Use prompt_toolkit for advanced prompt and autocompletion
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion

# mcp cli
from mcp_cli.tools.manager import ToolManager

# Use unified command system
from mcp_cli.adapters.interactive import (
    InteractiveCommandAdapter,
    InteractiveExitException,
)
from mcp_cli.commands import register_all_commands as register_unified_commands

# Keep old registry for now just for command name completion

# logger
logger = logging.getLogger(__name__)


class SlashCompleter(Completer):
    """Provides completions for slash commands based on registered commands."""

    def __init__(self, command_names: List[str]):
        self.command_names = command_names

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return
        token = text[1:]
        for name in self.command_names:
            if name.startswith(token):
                yield Completion(f"/{name}", start_position=-len(text))


async def interactive_mode(
    stream_manager: Any = None,
    tool_manager: Optional[ToolManager] = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    server_names: Optional[Dict[int, str]] = None,
    **kwargs,
) -> bool:
    """
    Launch the interactive mode CLI with slash-menu autocompletion.
    """

    # Register unified commands
    register_unified_commands()

    # Get command names for completion
    # TODO: Get from unified registry instead
    from mcp_cli.commands.registry import registry

    cmd_names = registry.get_command_names(include_aliases=True)

    # Intro panel
    print(
        Panel(
            Markdown(
                # "# Interactive Mode\n\n"
                "Type commands to interact with the system.\n"
                "Type 'help' to see available commands.\n"
                "Type 'exit' or 'quit' to exit.\n"
                "Type '/' to bring up the slash-menu."
            ),
            title="MCP Interactive Mode",
            style="bold cyan",
        )
    )

    # Initial help listing - use unified command
    await InteractiveCommandAdapter.handle_command("help")

    # Create a PromptSession with our completer
    session: PromptSession = PromptSession(
        completer=SlashCompleter(cmd_names),
        complete_while_typing=True,
    )

    # Main loop
    while True:
        try:
            raw = await asyncio.to_thread(session.prompt, "> ")
            line = raw.strip()

            # Skip empty
            if not line:
                continue

            # If user types a slash command exactly
            if line.startswith("/"):
                # strip leading slash and dispatch
                cmd_line = line[1:]
            else:
                # normal entry
                cmd_line = line

            # If line was just '/', show help
            if cmd_line == "":
                # Use unified help command
                await InteractiveCommandAdapter.handle_command("help")
                continue

            # Use unified command system
            try:
                # Pass the original command line to preserve quoting
                handled = await InteractiveCommandAdapter.handle_command(cmd_line)

                if not handled:
                    # Extract command name for error message
                    cmd_parts = cmd_line.split(maxsplit=1)
                    cmd_name = cmd_parts[0] if cmd_parts else cmd_line
                    print(f"[red]Unknown command: {cmd_name}[/red]")
                    print("[dim]Type 'help' to see available commands.[/dim]")
            except (KeyboardInterrupt, InteractiveExitException):
                # Exit requested
                return True

        except KeyboardInterrupt:
            print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except EOFError:
            print("\n[yellow]EOF detected. Exiting.[/yellow]")
            return True
        except Exception as e:
            logger.exception("Error in interactive mode")
            print(f"[red]Error: {e}[/red]")

    # This line is unreachable but needed for type checker
    # The while True loop above handles all exits
    # return True
