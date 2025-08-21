# mcp_cli/commands/help.py
"""
Help command for MCP CLI.

Displays help information for commands in both chat and CLI modes.
"""
from __future__ import annotations
from typing import Dict, Optional

from mcp_cli.ui import output, format_table

# Try interactive registry first, fall back to CLI registry
try:
    from mcp_cli.interactive.registry import InteractiveCommandRegistry as Registry
except ImportError:
    from mcp_cli.cli.registry import CommandRegistry as Registry


def help_action(command_name: Optional[str] = None) -> None:
    """
    Display help for a specific command or all commands.
    
    Args:
        command_name: Name of command to get help for. If None, shows all commands.
    """
    commands = _get_commands()
    
    if command_name:
        _show_command_help(command_name, commands)
    else:
        _show_all_commands(commands)


def _get_commands() -> Dict[str, object]:
    """Get available commands from the registry."""
    if hasattr(Registry, "get_all_commands"):
        return Registry.get_all_commands()
    return {}


def _show_command_help(command_name: str, commands: Dict[str, object]) -> None:
    """Show detailed help for a specific command."""
    cmd = commands.get(command_name)
    
    if cmd is None:
        output.error(f"Unknown command: {command_name}")
        return
    
    # Display command details
    help_text = cmd.help or "No description provided."
    
    output.panel(
        f"## {cmd.name}\n\n{help_text}",
        title="Command Help",
        style="cyan"
    )
    
    # Show aliases if available
    if hasattr(cmd, "aliases") and cmd.aliases:
        output.print(f"\n[dim]Aliases: {', '.join(cmd.aliases)}[/dim]")


def _show_all_commands(commands: Dict[str, object]) -> None:
    """Show a summary table of all available commands."""
    if not commands:
        output.warning("No commands available")
        return
    
    # Build table data
    table_data = []
    for name, cmd in sorted(commands.items()):
        # Extract first meaningful line from help text
        desc = _extract_description(cmd.help)
        
        # Get aliases
        aliases = "-"
        if hasattr(cmd, "aliases") and cmd.aliases:
            aliases = ", ".join(cmd.aliases)
        
        table_data.append({
            "Command": name,
            "Aliases": aliases,
            "Description": desc
        })
    
    # Display table
    table = format_table(
        table_data,
        title="Available Commands",
        columns=["Command", "Aliases", "Description"]
    )
    output.print_table(table)
    
    output.hint("\nType 'help <command>' for detailed information on a specific command.")


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