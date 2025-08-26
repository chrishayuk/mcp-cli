# mcp_cli/commands/help.py
"""
Help command for MCP CLI.

Displays help information for commands in both chat and CLI modes.
"""

from __future__ import annotations
from typing import Dict, Optional, Any

from chuk_term.ui import output, format_table

# Try interactive registry first, fall back to CLI registry
try:
    from mcp_cli.interactive.registry import InteractiveCommandRegistry as Registry
except ImportError:
    from mcp_cli.cli.registry import CommandRegistry as Registry


def help_action(command_name: Optional[str] = None, console: Any = None) -> None:
    """
    Display help for a specific command or all commands.

    Args:
        command_name: Name of command to get help for. If None, shows all commands.
        console: Rich console object (optional, for compatibility with interactive mode)
    """
    # Note: console argument is accepted for backward compatibility but not used
    # The new implementation uses the UI output module instead

    commands = _get_commands()

    if command_name:
        _show_command_help(command_name, commands)
    else:
        _show_all_commands(commands)


def _get_commands() -> Dict[str, object]:
    """Get available commands from the registry."""
    if hasattr(Registry, "get_all_commands"):
        result = Registry.get_all_commands()
        # CLI registry returns a list, Interactive registry returns a dict
        if isinstance(result, list):
            # Convert list to dict using command.name as key
            return {getattr(cmd, "name", str(i)): cmd for i, cmd in enumerate(result)}
        return result
    elif hasattr(Registry, "_commands"):
        return Registry._commands
    return {}


def _show_command_help(command_name: str, commands: Dict[str, object]) -> None:
    """Show detailed help for a specific command."""
    cmd = commands.get(command_name)

    if cmd is None:
        output.error(f"Unknown command: {command_name}")
        return

    # Get help text
    help_text = getattr(cmd, "help", "No description provided.")

    # Display command details
    cmd_name = getattr(cmd, "name", command_name)

    output.panel(f"## {cmd_name}\n\n{help_text}", title="Command Help", style="cyan")

    # Show aliases if available
    aliases = getattr(cmd, "aliases", [])
    if aliases:
        output.print(f"\n[dim]Aliases: {', '.join(aliases)}[/dim]")


def _show_all_commands(commands: Dict[str, object]) -> None:
    """Show a summary table of all available commands."""
    if not commands:
        output.warning("No commands available")
        return

    # Build table data
    table_data = []
    for name, cmd in sorted(commands.items()):
        # Get help text
        help_text = getattr(cmd, "help", "")

        # Extract first meaningful line from help text
        desc = _extract_description(help_text)

        # Get aliases
        aliases = "-"
        cmd_aliases = getattr(cmd, "aliases", [])
        if cmd_aliases:
            aliases = ", ".join(cmd_aliases)

        table_data.append({"Command": name, "Aliases": aliases, "Description": desc})

    # Display table
    table = format_table(
        table_data,
        title="Available Commands",
        columns=["Command", "Aliases", "Description"],
    )
    output.print_table(table)

    output.hint(
        "\nType 'help <command>' for detailed information on a specific command."
    )


def _extract_description(help_text: Optional[str]) -> str:
    """Extract a one-line description from help text."""
    if not help_text:
        return "No description"

    # Find first non-empty line that doesn't start with "usage"
    for line in help_text.splitlines():
        line = line.strip()
        if line and not line.lower().startswith("usage"):
            return line

    return "No description"
