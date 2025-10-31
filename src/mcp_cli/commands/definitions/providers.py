# src/mcp_cli/commands/definitions/provider.py
"""
Unified provider command implementation.
Uses the existing enhanced provider commands from mcp_cli.commands.provider
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandGroup,
    CommandParameter,
    CommandResult,
)


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
        return []  # Remove provider alias - it's now its own command

    @property
    def description(self) -> str:
        return "List available LLM providers"

    @property
    def help_text(self) -> str:
        return """
Manage LLM providers for the MCP CLI.

Subcommands:
  list    - List all available providers
  custom  - List custom OpenAI-compatible providers
  add     - Add a custom provider
  remove  - Remove a custom provider
  set     - Configure provider settings
  show    - Show current provider status

Custom Provider Management:
  /provider add <name> <api_base> [models...]
    Add a custom OpenAI-compatible provider (LocalAI, proxies, etc.)
    
  /provider remove <name>
    Remove a custom provider
    
  /provider custom
    List all custom providers

Usage:
  /provider              - Show current provider status
  /providers             - List all providers (preferred)
  /provider <name>       - Switch to a different provider
  /provider list         - List all providers
  
Examples:
  # Switch providers
  /provider ollama       - Switch to Ollama provider
  /provider openai       - Switch to OpenAI provider
  
  # Add custom providers
  /provider add localai http://localhost:8080/v1 gpt-4 gpt-3.5-turbo
  /provider add myproxy https://proxy.example.com/v1 custom-model
  
  # Use custom provider (after setting API key)
  export LOCALAI_API_KEY=your-key
  /provider localai
  
  # Remove custom provider
  /provider remove localai

Note: API keys are NEVER stored in config. Use environment variables:
  Pattern: {PROVIDER_NAME}_API_KEY
  Example: LOCALAI_API_KEY, MYPROXY_API_KEY
"""

    async def execute(self, subcommand: str | None = None, **kwargs) -> CommandResult:
        """Execute the provider command - handle direct provider switching."""
        from mcp_cli.commands.actions.providers import provider_action_async

        # Check if we have args (could be provider name or subcommand)
        args = kwargs.get("args", [])

        if not args:
            # No arguments - behavior depends on which command was used
            # /providers (plural) -> list all providers
            # /provider (singular) -> show current status

            # Try to determine which command was used (this is a heuristic)
            # In chat context, we might not have this info, so default to status for /provider
            # and list for /providers based on the primary command name

            # Since this is the ProviderCommand class with name="providers",
            # when called without args as /providers, show the list
            # When called as /provider (alias), it will still come here but we default to list
            # for consistency with /models behavior

            try:
                # Default to list when no arguments (like /models does)
                from mcp_cli.commands.models import ProviderActionParams

                params = ProviderActionParams(args=["list"])
                await provider_action_async(params)
                return CommandResult(success=True)
            except Exception as e:
                return CommandResult(
                    success=False, error=f"Failed to list providers: {str(e)}"
                )

        # Check if the first arg is a known subcommand
        first_arg = args[0] if isinstance(args, list) else str(args)

        # Known subcommands that should be handled by subcommand classes
        if first_arg.lower() in [
            "list",
            "ls",
            "set",
            "use",
            "switch",
            "show",
            "current",
            "status",
        ]:
            # Let the parent class handle the subcommand routing
            return await super().execute(**kwargs)

        # Otherwise, treat it as a provider name to switch to
        try:
            # Pass the provider name directly to switch
            from mcp_cli.commands.models import ProviderActionParams

            if isinstance(args, list):
                params = ProviderActionParams(args=args)
            else:
                params = ProviderActionParams(args=[str(args)])
            await provider_action_async(params)
            return CommandResult(success=True)
        except Exception as e:
            return CommandResult(
                success=False, error=f"Failed to switch provider: {str(e)}"
            )


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
        from mcp_cli.commands.actions.providers import provider_action_async

        try:
            # Use the existing enhanced implementation
            # It handles all the display internally with rich formatting
            # Pass "list" as the command
            from mcp_cli.commands.models import ProviderActionParams

            params = ProviderActionParams(args=["list"])
            await provider_action_async(params)

            # The existing implementation handles all output directly
            # Just return success
            return CommandResult(success=True, data={"command": "provider list"})

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
        from mcp_cli.commands.actions.providers import provider_action_async

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
            from mcp_cli.commands.models import ProviderActionParams

            params = ProviderActionParams(args=[provider_name])
            await provider_action_async(params)

            # The existing implementation handles all output directly
            return CommandResult(success=True, data={"provider": provider_name})

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
        from mcp_cli.commands.actions.providers import provider_action_async

        try:
            from mcp_cli.commands.models import ProviderActionParams

            # Use the existing enhanced implementation
            # Pass no arguments to show current status
            params = ProviderActionParams(args=[])
            await provider_action_async(params)

            # The existing implementation handles all output directly
            return CommandResult(success=True, data={"command": "provider show"})

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to get provider info: {str(e)}",
            )
