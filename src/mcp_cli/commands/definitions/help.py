# src/mcp_cli/commands/definitions/help.py
"""
Unified help command implementation.
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from mcp_cli.commands.registry import UnifiedCommandRegistry
from chuk_term.ui import format_table, output


class HelpCommand(UnifiedCommand):
    """Show help information for commands."""

    @property
    def name(self) -> str:
        return "help"

    @property
    def aliases(self) -> List[str]:
        return ["h", "?"]

    @property
    def description(self) -> str:
        return "Show help information for commands"

    @property
    def help_text(self) -> str:
        return """
Show help information for available commands.

Usage:
  /help [command]     - Show help (chat mode)
  help [command]      - Show help (interactive mode)
  mcp-cli help        - Show help (CLI mode)
  
Examples:
  /help              - List all commands
  /help servers      - Show detailed help for servers command
  help tools         - Show help for tools command
"""

    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="command",
                type=str,
                required=False,
                help="Command to get help for",
            ),
        ]

    @property
    def requires_context(self) -> bool:
        """Help doesn't need tool manager context."""
        return False

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the help command."""
        command_name = kwargs.get("command") or kwargs.get("args")

        # Handle list arguments
        if isinstance(command_name, list):
            command_name = command_name[0] if command_name else None

        # Get the registry singleton instance
        registry = UnifiedCommandRegistry()

        # Determine which mode we're in based on context
        mode = kwargs.get("mode", CommandMode.CHAT)

        try:
            if command_name:
                # Show help for specific command
                command = registry.get(command_name, mode=mode)
                if not command:
                    return CommandResult(
                        success=False,
                        error=f"Unknown command: {command_name}",
                    )

                # Display command help directly
                help_content = (
                    f"## {command.name}\n\n{command.help_text or command.description}"
                )
                output.panel(
                    help_content,
                    title="Command Help",
                    style="cyan",
                )

                if command.aliases:
                    output.print(f"\n[dim]Aliases: {', '.join(command.aliases)}[/dim]")
                    help_content += f"\n\nAliases: {', '.join(command.aliases)}"

                return CommandResult(success=True, output=help_content)

            else:
                # List all available commands
                commands = registry.list_commands(mode=mode)

                if not commands:
                    return CommandResult(
                        success=True,
                        output="No commands available.",
                    )

                # Format as table
                table_data = []
                for cmd in commands:
                    # Check if this is a command group with subcommands
                    from mcp_cli.commands.base import CommandGroup

                    has_subcommands = (
                        isinstance(cmd, CommandGroup)
                        and hasattr(cmd, "subcommands")
                        and cmd.subcommands
                    )

                    # Format command name with indicator for subcommands
                    command_display = cmd.name
                    if has_subcommands:
                        command_display = f"{cmd.name} ▸"

                    row = {
                        "Command": command_display,
                        "Description": cmd.description,
                    }
                    if cmd.aliases:
                        row["Aliases"] = ", ".join(cmd.aliases)

                    # Add subcommands info if available
                    if has_subcommands and isinstance(cmd, CommandGroup):
                        subcommand_names = list(cmd.subcommands.keys())
                        if len(subcommand_names) <= 3:
                            row["Subcommands"] = ", ".join(subcommand_names)
                        else:
                            row["Subcommands"] = (
                                f"{', '.join(subcommand_names[:3])}, ..."
                            )
                    else:
                        row["Subcommands"] = ""

                    table_data.append(row)

                # Determine columns to show
                has_aliases = any(cmd.aliases for cmd in commands)
                has_subs = any(row.get("Subcommands") for row in table_data)

                columns = ["Command"]
                if has_aliases:
                    columns.append("Aliases")
                if has_subs:
                    columns.append("Subcommands")
                columns.append("Description")

                table = format_table(
                    table_data,
                    title="Available Commands",
                    columns=columns,
                )

                # Display table directly
                output.print_table(table)

                # Show helpful hints
                hints = []
                hints.append("Type '/help <command>' for detailed information")
                if has_subs:
                    hints.append("Commands with ▸ have subcommands")
                    hints.append("Try '/help <command>' to see subcommands")

                output.hint("\n" + " | ".join(hints))

                # Build output string for result
                command_names = [cmd.name for cmd in commands]
                output_str = f"Available Commands ({len(commands)} commands): {', '.join(command_names)}"
                return CommandResult(success=True, output=output_str)

        except Exception as e:
            return CommandResult(success=False, error=f"Failed to show help: {str(e)}")
