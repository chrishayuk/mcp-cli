# src/mcp_cli/interactive/shell.py
"""Interactive shell implementation for MCP CLI with slash-menu autocompletion."""

from __future__ import annotations
import asyncio
import logging
import shlex
from typing import Any, Dict, List, Optional

from rich import print
from chuk_term.ui import output
from rich.markdown import Markdown
from rich.panel import Panel

# Use prompt_toolkit for advanced prompt and autocompletion
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion

# mcp cli
from mcp_cli.tools.manager import ToolManager

# commands
from mcp_cli.interactive.commands import register_all_commands
from mcp_cli.interactive.registry import InteractiveCommandRegistry

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
    console = output._console

    # Register commands
    register_all_commands()
    cmd_names = list(InteractiveCommandRegistry.get_all_commands().keys())

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

    # Initial help listing
    help_cmd = InteractiveCommandRegistry.get_command("help")
    if help_cmd:
        await help_cmd.execute([], tool_manager, server_names=server_names)

    # Create a PromptSession with our completer
    session = PromptSession(
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
                if help_cmd:
                    await help_cmd.execute([], tool_manager, server_names=server_names)
                continue

            # Parse
            try:
                parts = shlex.split(cmd_line)
            except ValueError:
                parts = cmd_line.split()

            cmd_name = parts[0].lower()
            args = parts[1:]

            # Lookup and execute
            cmd = InteractiveCommandRegistry.get_command(cmd_name)
            if cmd:
                result = await cmd.execute(
                    args, tool_manager, server_names=server_names, **kwargs
                )
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
