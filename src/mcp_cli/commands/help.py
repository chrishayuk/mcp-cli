# mcp_cli/commands/help.py
"""
Shared help logic for both interactive and CLI interfaces.
"""
from typing import List, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# registry
from mcp_cli.interactive.registry import InteractiveCommandRegistry


def help_action(
    console: Console,
    command_name: Optional[str] = None
) -> None:
    """
    Display help for all commands, or help for a specific command
    if command_name is provided.
    """
    commands = InteractiveCommandRegistry.get_all_commands()

    if command_name and command_name in commands:
        cmd = InteractiveCommandRegistry.get_command(command_name)
        console.print(Panel(
            Markdown(f"## Command: {cmd.name}\n\n{cmd.help or ''}"),
            title="Command Help",
            style="cyan"
        ))
        if cmd.aliases:
            aliases_str = ", ".join(cmd.aliases)
            console.print(f"[dim]Aliases: {aliases_str}[/dim]")
        return

    # Otherwise, show full list
    table = Table(title="Available Commands")
    table.add_column("Command", style="green")
    table.add_column("Description")

    for name, cmd in sorted(commands.items()):
        help_text = (cmd.help or "").split("\n")[0]
        if len(help_text) > 60:
            help_text = help_text[:57] + "..."
        table.add_row(name, help_text)

    console.print(table)
    console.print("[dim]Type 'help <command>' for more information about a specific command.[/dim]")
