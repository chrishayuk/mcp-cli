# src/mcp_cli/commands/impl/help.py
"""
Unified help command implementation.
"""

from __future__ import annotations

from typing import List, Optional

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
                output.panel(
                    f"## {command.name}\n\n{command.help_text or command.description}",
                    title="Command Help",
                    style="cyan"
                )
                
                if command.aliases:
                    output.print(f"\n[dim]Aliases: {', '.join(command.aliases)}[/dim]")
                
                return CommandResult(success=True)
            
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
                    row = {
                        "Command": cmd.name,
                        "Description": cmd.description,
                    }
                    if cmd.aliases:
                        row["Aliases"] = ", ".join(cmd.aliases)
                    table_data.append(row)
                
                columns = ["Command", "Aliases", "Description"] if any(cmd.aliases for cmd in commands) else ["Command", "Description"]
                
                table = format_table(
                    table_data,
                    title="Available Commands",
                    columns=columns,
                )
                
                # Display table directly
                output.print_table(table)
                output.hint("\nType 'help <command>' for detailed information on a specific command.")
                
                return CommandResult(success=True)
                
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to show help: {str(e)}"
            )