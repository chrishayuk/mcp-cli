# src/mcp_cli/commands/actions/help.py
"""
Help command for MCP CLI.

Displays help information for commands in both chat and CLI modes.
"""

from __future__ import annotations
from typing import Dict, Optional, Any

from chuk_term.ui import output, format_table
from mcp_cli.commands.registry import registry


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
    """Get available commands from the unified registry."""
    commands: Dict[str, object] = {}
    for cmd in registry.list_commands():
        commands[cmd.name] = cmd
    return commands


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

    # Add provider management tips
    output.print()
    output.tip("💡 Provider Management:")
    output.info("  • List all: /providers")
    output.info("  • Add custom: /provider add <name> <api_base> [model]")
    output.info("  • Switch: /provider <name>")
    output.info("  • Remove: /provider remove <name>")
    output.print()
    output.hint("Custom providers need API keys as environment variables:")
    output.info("  Pattern: {PROVIDER_NAME}_API_KEY")
    output.info("  Example: 'localai' → export LOCALAI_API_KEY=your-key")
    output.info("  Example: 'my-llm' → export MY_LLM_API_KEY=your-key")


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
