# src/mcp_cli/commands/impl/provider.py
"""
Unified provider command implementation.
Uses the existing enhanced provider commands from mcp_cli.commands.provider
"""

from __future__ import annotations

from typing import List, Optional

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandGroup,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from mcp_cli.context import get_context
from chuk_term.ui import output, format_table


class ProviderCommand(CommandGroup):
    """Provider command group."""
    
    def __init__(self):
        super().__init__()
        # Add subcommands
        self.add_subcommand(ProviderListCommand())
        self.add_subcommand(ProviderSetCommand())
        self.add_subcommand(ProviderShowCommand())
    
    @property
    def name(self) -> str:
        return "providers"
    
    @property
    def aliases(self) -> List[str]:
        return []
    
    @property
    def description(self) -> str:
        return "Manage LLM providers"
    
    @property
    def help_text(self) -> str:
        return """
Manage LLM providers for the MCP CLI.

Subcommands:
  list  - List available providers
  set   - Set the active provider
  show  - Show current provider

Usage:
  /providers list          - List all providers (chat mode)
  providers list           - List all providers (interactive mode)
  mcp-cli providers list   - List all providers (CLI mode)
  
Examples:
  /providers set openai
  /providers show
  /providers list
"""


class ProviderListCommand(UnifiedCommand):
    """List available providers."""
    
    @property
    def name(self) -> str:
        return "list"
    
    @property
    def aliases(self) -> List[str]:
        return ["ls"]
    
    @property
    def description(self) -> str:
        return "List all available LLM providers"
    
    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="detailed",
                type=bool,
                default=False,
                help="Show detailed provider information",
                is_flag=True,
            ),
        ]
    
    async def execute(self, **kwargs) -> CommandResult:
        """Execute the provider list command."""
        # Import the existing provider implementation
        from mcp_cli.commands.provider import provider_action_async
        
        try:
            # Use the existing enhanced implementation
            # It handles all the display internally with rich formatting
            # Pass "list" as the command
            await provider_action_async(["list"])
            
            # The existing implementation handles all output directly
            # Just return success
            return CommandResult(
                success=True,
                data={"command": "provider list"}
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list providers: {str(e)}",
            )


class ProviderSetCommand(UnifiedCommand):
    """Set the active provider."""
    
    @property
    def name(self) -> str:
        return "set"
    
    @property
    def aliases(self) -> List[str]:
        return ["use", "switch"]
    
    @property
    def description(self) -> str:
        return "Set the active LLM provider"
    
    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="provider_name",
                type=str,
                required=True,
                help="Name of the provider to set",
            ),
        ]
    
    async def execute(self, **kwargs) -> CommandResult:
        """Execute the provider set command."""
        # Import the existing provider implementation
        from mcp_cli.commands.provider import provider_action_async
        
        # Get provider name
        provider_name = kwargs.get("provider_name")
        if not provider_name and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list):
                provider_name = args_val[0] if args_val else None
            elif isinstance(args_val, str):
                provider_name = args_val
        
        if not provider_name:
            return CommandResult(
                success=False,
                error="Provider name is required. Usage: /provider set <name>",
            )
        
        try:
            # Use the existing enhanced implementation
            # Pass the provider name directly to switch to it
            await provider_action_async([provider_name])
            
            # The existing implementation handles all output directly
            return CommandResult(
                success=True,
                data={"provider": provider_name}
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to set provider: {str(e)}",
            )


class ProviderShowCommand(UnifiedCommand):
    """Show current provider."""
    
    @property
    def name(self) -> str:
        return "show"
    
    @property
    def aliases(self) -> List[str]:
        return ["current", "status"]
    
    @property
    def description(self) -> str:
        return "Show the current active provider"
    
    async def execute(self, **kwargs) -> CommandResult:
        """Execute the provider show command."""
        # Import the existing provider implementation
        from mcp_cli.commands.provider import provider_action_async
        
        try:
            # Use the existing enhanced implementation
            # Pass no arguments to show current status
            await provider_action_async([])
            
            # The existing implementation handles all output directly
            return CommandResult(
                success=True,
                data={"command": "provider show"}
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to get provider info: {str(e)}",
            )