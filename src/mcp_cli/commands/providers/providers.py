# src/mcp_cli/commands/definitions/provider.py
"""
Unified provider command implementation.
Uses the existing enhanced provider commands from mcp_cli.commands.provider
"""

from __future__ import annotations


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
    def aliases(self) -> list[str]:
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
        from mcp_cli.context import get_context
        from chuk_term.ui import output, format_table

        # Check if we have args (could be provider name or subcommand)
        args = kwargs.get("args", [])

        if not args:
            # No arguments - list all providers
            try:
                context = get_context()
                if not context or not context.llm_manager:
                    return CommandResult(
                        success=False, error="No LLM manager available."
                    )

                providers = context.llm_manager.list_providers()
                current_provider = context.llm_manager.get_current_provider()

                # Build table data
                table_data = []
                for provider in providers:
                    is_current = "✓" if provider == current_provider else ""
                    table_data.append({"Current": is_current, "Provider": provider})

                # Display table
                table = format_table(
                    table_data,
                    title=f"{len(providers)} Available Providers",
                    columns=["Current", "Provider"],
                )
                output.print_table(table)

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
            context = get_context()
            if not context or not context.llm_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            provider_name = first_arg
            context.llm_manager.set_provider(provider_name)
            output.success(f"Switched to provider: {provider_name}")

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
    def aliases(self) -> list[str]:
        return ["ls"]

    @property
    def description(self) -> str:
        return "List all available LLM providers"

    @property
    def parameters(self) -> list[CommandParameter]:
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
        from mcp_cli.context import get_context
        from chuk_term.ui import output, format_table

        try:
            context = get_context()
            if not context or not context.llm_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            providers = context.llm_manager.list_providers()
            current_provider = context.llm_manager.get_current_provider()

            # Build table data
            table_data = []
            for provider in providers:
                is_current = "✓" if provider == current_provider else ""
                table_data.append({"Current": is_current, "Provider": provider})

            # Display table
            table = format_table(
                table_data,
                title=f"{len(providers)} Available Providers",
                columns=["Current", "Provider"],
            )
            output.print_table(table)

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
    def aliases(self) -> list[str]:
        return ["use", "switch"]

    @property
    def description(self) -> str:
        return "Set the active LLM provider"

    @property
    def parameters(self) -> list[CommandParameter]:
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
        from mcp_cli.context import get_context
        from chuk_term.ui import output

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
            context = get_context()
            if not context or not context.llm_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            context.llm_manager.set_provider(provider_name)
            output.success(f"Switched to provider: {provider_name}")

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
    def aliases(self) -> list[str]:
        return ["current", "status"]

    @property
    def description(self) -> str:
        return "Show the current active provider"

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the provider show command."""
        from mcp_cli.context import get_context
        from chuk_term.ui import output

        try:
            context = get_context()
            if not context or not context.llm_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            current_provider = context.llm_manager.get_current_provider()
            current_model = context.llm_manager.get_current_model()

            output.panel(
                f"Provider: {current_provider}\nModel: {current_model}",
                title="Current Provider",
            )

            return CommandResult(success=True, data={"command": "provider show"})

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to get provider info: {str(e)}",
            )
