# mcp_cli/interactive/commands/provider.py
"""
Interactive **provider** and **providers** commands - inspect or switch the active LLM provider
(and optionally the default model) from inside the shell.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from chuk_term.ui import output
from mcp_cli.commands.provider import provider_action_async
from .base import InteractiveCommand

log = logging.getLogger(__name__)


class ProviderCommand(InteractiveCommand):
    """Show / switch providers, tweak config, run diagnostics."""

    def __init__(self) -> None:
        super().__init__(
            name="provider",
            aliases=["p"],
            help_text=(
                "Manage LLM providers.\n\n"
                "  provider                          Show current provider/model\n"
                "  provider list                     List available providers\n"
                "  provider config                   Show provider configuration\n"
                "  provider diagnostic [prov]        Probe provider(s) health\n"
                "  provider set <prov> <key> <val>   Update one config key\n"
                "  provider <prov> [model]           Switch provider (and model)\n"
                "\nExamples:\n"
                "  provider list                     # See all providers with model counts\n"
                "  provider anthropic                # Switch to Anthropic with default model\n"
                "  provider openai gpt-4o            # Switch to OpenAI with specific model\n"
                "  provider diagnostic openai        # Check OpenAI setup\n"
                "  provider set openai api_key sk-.. # Configure API key\n"
            ),
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: Any = None,
        **ctx: Dict[str, Any],
    ) -> None:
        """
        Delegate to :func:`provider_action_async`.

        *args* arrive without the leading command word, exactly as the
        shared helper expects.
        """

        # The provider command does not require ToolManager, but log if absent
        if tool_manager is None:
            log.debug("ProviderCommand executed without ToolManager – OK for now.")

        # Ensure we have a model_manager in context
        if "model_manager" not in ctx:
            log.debug("Creating ModelManager for interactive provider command")
            from mcp_cli.model_manager import ModelManager

            ctx["model_manager"] = ModelManager()

        try:
            await provider_action_async(args, context=ctx)
        except Exception as exc:  # noqa: BLE001
            output.print(f"[red]Provider command failed:[/red] {exc}")
            log.exception("ProviderCommand error")

            # Provide helpful debugging info if the error seems related to our recent fixes
            if "available_models" in str(exc) or "models" in str(exc):
                output.print(
                    "[yellow]Hint:[/yellow] This might be a chuk-llm 0.7 compatibility issue."
                )
                output.print(
                    "[yellow]Try:[/yellow] Ensure you're using the latest ModelManager fixes."
                )

    def get_completions(self, partial_command: str) -> List[str]:
        """Provide tab completion for provider commands."""
        try:
            from mcp_cli.model_manager import ModelManager

            words = partial_command.strip().split()

            # If just "provider" or "p", suggest subcommands
            if len(words) <= 1:
                return ["list", "config", "diagnostic", "set"]

            subcommand = words[1].lower()

            # Provider name completions
            if subcommand in ["diagnostic", "set"] or (
                subcommand not in ["list", "config"]
            ):
                mm = ModelManager()
                providers = mm.list_providers()

                if len(words) == 2:
                    # Complete provider names
                    return [p for p in providers if p.startswith(words[1])]
                elif len(words) == 3 and subcommand not in ["diagnostic"]:
                    # For "provider <provider> <model>" or "provider set <provider> <key>"
                    if subcommand == "set":
                        return ["api_key", "api_base", "default_model"]
                    else:
                        # Complete model names for the specified provider
                        try:
                            provider = words[1]
                            models = mm.get_available_models(provider)
                            return [m for m in models if m.startswith(words[2])]
                        except:
                            return []

            return []

        except Exception as e:
            log.debug(f"Completion error: {e}")
            return []


class ProvidersCommand(InteractiveCommand):
    """List providers (plural form - defaults to list)."""

    def __init__(self) -> None:
        super().__init__(
            name="providers",
            aliases=["ps"],
            help_text=(
                "List and manage LLM providers (defaults to list).\n\n"
                "  providers                         List all available providers\n"
                "  providers list                    List available providers (explicit)\n"
                "  providers config                  Show provider configuration\n"
                "  providers diagnostic [prov]       Probe provider(s) health\n"
                "  providers set <prov> <key> <val>  Update one config key\n"
                "  providers <prov> [model]          Switch provider (and model)\n"
                "\nExamples:\n"
                "  providers                         # List all providers\n"
                "  providers anthropic               # Switch to Anthropic\n"
                "  providers openai gpt-4o           # Switch to OpenAI with specific model\n"
            ),
        )

    async def execute(
        self,
        args: List[str],
        tool_manager: Any = None,
        **ctx: Dict[str, Any],
    ) -> None:
        """
        Delegate to :func:`provider_action_async` with default to list.
        """

        if tool_manager is None:
            log.debug("ProvidersCommand executed without ToolManager – OK for now.")

        # Ensure we have a model_manager in context
        if "model_manager" not in ctx:
            log.debug("Creating ModelManager for interactive providers command")
            from mcp_cli.model_manager import ModelManager

            ctx["model_manager"] = ModelManager()

        try:
            # If no arguments provided, default to "list"
            if not args:
                args = ["list"]

            await provider_action_async(args, context=ctx)
        except Exception as exc:  # noqa: BLE001
            output.print(f"[red]Providers command failed:[/red] {exc}")
            log.exception("ProvidersCommand error")

    def get_completions(self, partial_command: str) -> List[str]:
        """Provide tab completion for providers commands."""
        try:
            from mcp_cli.model_manager import ModelManager

            words = partial_command.strip().split()

            # If just "providers" or "ps", suggest subcommands and provider names
            if len(words) <= 1:
                # Return both subcommands and provider names
                subcommands = ["list", "config", "diagnostic", "set"]
                mm = ModelManager()
                providers = mm.list_providers()
                return subcommands + providers

            subcommand = words[1].lower()

            # Provider name completions
            if subcommand in ["diagnostic", "set"] or (
                subcommand not in ["list", "config"]
            ):
                mm = ModelManager()
                providers = mm.list_providers()

                if len(words) == 2:
                    # Complete provider names
                    return [p for p in providers if p.startswith(words[1])]
                elif len(words) == 3 and subcommand not in ["diagnostic"]:
                    # For "providers <provider> <model>" or "providers set <provider> <key>"
                    if subcommand == "set":
                        return ["api_key", "api_base", "default_model"]
                    else:
                        # Complete model names for the specified provider
                        try:
                            provider = words[1]
                            models = mm.get_available_models(provider)
                            return [m for m in models if m.startswith(words[2])]
                        except:
                            return []

            return []

        except Exception as e:
            log.debug(f"Completion error: {e}")
            return []
