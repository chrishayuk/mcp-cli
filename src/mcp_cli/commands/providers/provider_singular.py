# src/mcp_cli/commands/definitions/provider_singular.py
"""
Singular provider command - shows current status.
"""

from __future__ import annotations


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
    def aliases(self) -> list[str]:
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
        from mcp_cli.context import get_context
        from chuk_term.ui import output

        # Get args
        args = kwargs.get("args", [])

        if not args:
            # No arguments - show current status (singular behavior)
            try:
                context = get_context()
                if not context or not context.llm_manager:
                    return CommandResult(
                        success=False, error="No LLM manager available."
                    )

                current_provider = context.llm_manager.get_current_provider()
                current_model = context.llm_manager.get_current_model()

                output.panel(
                    f"Provider: {current_provider}\nModel: {current_model}",
                    title="Current Provider Status",
                )
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
                # These should be handled by the providers command group
                return CommandResult(
                    success=False,
                    error=f"Use /providers {first_arg} for this command",
                )
            else:
                # Treat as provider name to switch to
                try:
                    context = get_context()
                    if not context or not context.llm_manager:
                        return CommandResult(
                            success=False, error="No LLM manager available."
                        )

                    provider_name = first_arg
                    context.llm_manager.set_provider(provider_name)
                    output.success(f"Switched to provider: {provider_name}")

                    return CommandResult(success=True)
                except Exception as e:
                    return CommandResult(
                        success=False, error=f"Failed to switch provider: {str(e)}"
                    )
