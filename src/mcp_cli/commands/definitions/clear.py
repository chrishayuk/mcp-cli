# src/mcp_cli/commands/definitions/clear.py
"""
Unified clear command implementation.
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandResult,
)


class ClearCommand(UnifiedCommand):
    """Clear the terminal screen."""
    
    @property
    def name(self) -> str:
        return "clear"
    
    @property
    def aliases(self) -> List[str]:
        return []
    
    @property
    def description(self) -> str:
        return "Clear the terminal screen"
    
    @property
    def help_text(self) -> str:
        return """
Clear the terminal screen.

Usage:
  /clear    - Clear screen (chat mode)
  clear     - Clear screen (interactive mode)
  
Aliases: cls
"""
    
    @property
    def modes(self) -> CommandMode:
        """Clear is only for chat and interactive modes."""
        return CommandMode.CHAT | CommandMode.INTERACTIVE
    
    @property
    def requires_context(self) -> bool:
        """Clear doesn't need tool manager context."""
        return False
    
    async def execute(self, **kwargs) -> CommandResult:
        """Execute the clear command."""
        # Import the clear action
        from mcp_cli.commands.actions.clear import clear_action
        
        # Execute the action
        verbose = kwargs.get("verbose", False)
        clear_action(verbose=verbose)
        
        return CommandResult(
            success=True,
            should_clear=True,  # Signal that screen was cleared
        )