# src/mcp_cli/commands/definitions/provider_singular.py
"""
Singular provider command - shows current status.
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandResult,
)


class ProviderSingularCommand(UnifiedCommand):
    """Show current provider status."""

    @property
    def name(self) -> str:
        return "provider"

    @property
    def aliases(self) -> List[str]:
        return []  # No aliases for singular form

    @property
    def description(self) -> str:
        return "Show current provider status or switch providers"

    @property
    def help_text(self) -> str:
        return """
Show current LLM provider status or switch to a different provider.

Usage:
  /provider              - Show current provider status
  /provider <name>       - Switch to a different provider
  
Examples:
  /provider              - Show current status
  /provider ollama       - Switch to Ollama
  /provider openai       - Switch to OpenAI
"""

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the provider command."""
        from mcp_cli.commands.actions.providers import provider_action_async

        # Get args
        args = kwargs.get("args", [])

        if not args:
            # No arguments - show current status (singular behavior)
            try:
                from mcp_cli.commands.models import ProviderActionParams

                params = ProviderActionParams(args=[])
                await provider_action_async(params)  # Empty args = show status
                return CommandResult(success=True)
            except Exception as e:
                return CommandResult(
                    success=False, error=f"Failed to show provider status: {str(e)}"
                )
        else:
            # Has arguments - could be provider name to switch to
            first_arg = args[0] if isinstance(args, list) else str(args)

            # If it's a known subcommand, handle it
            if first_arg.lower() in ["list", "ls", "set"]:
                # Delegate to the action
                try:
                    from mcp_cli.commands.models import ProviderActionParams

                    args_list = args if isinstance(args, list) else [str(args)]
                    params = ProviderActionParams(args=args_list)
                    await provider_action_async(params)
                    return CommandResult(success=True)
                except Exception as e:
                    return CommandResult(
                        success=False, error=f"Command failed: {str(e)}"
                    )
            else:
                # Treat as provider name to switch to
                try:
                    from mcp_cli.commands.models import ProviderActionParams

                    args_list = args if isinstance(args, list) else [str(args)]
                    params = ProviderActionParams(args=args_list)
                    await provider_action_async(params)

                    return CommandResult(success=True)
                except Exception as e:
                    return CommandResult(
                        success=False, error=f"Failed to switch provider: {str(e)}"
                    )
