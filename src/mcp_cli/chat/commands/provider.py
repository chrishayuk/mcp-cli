# mcp_cli/chat/commands/provider.py
"""
Chat-mode `/provider` and `/providers` commands for MCP-CLI
========================================

Gives you full control over **LLM providers** without leaving the chat
session.

At a glance
-----------
* `/provider`                      - show current provider & model
* `/provider list`                 - list available providers
* `/providers`                     - list available providers (shortcut)
* `/providers`                     - list available providers (shortcut)
* `/provider config`               - dump full provider configs
* `/provider diagnostic`           - ping each provider with a tiny prompt
* `/provider set <prov> <k> <v>`   - change one config value (e.g. API key)
* `/provider <prov>  [model]`      - switch provider (and optional model)

All heavy lifting is delegated to
:meth:`mcp_cli.commands.provider.provider_action_async`, which performs
safety probes before committing any switch.
"""

from __future__ import annotations
import logging
from typing import List

# Cross-platform chuk_term console helper
from chuk_term.ui import output
from chuk_term.ui.prompts import confirm

# Shared implementation
from mcp_cli.commands.provider import provider_action_async
from mcp_cli.chat.commands import register_command
from mcp_cli.context import get_context

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# /provider entry-point
# ════════════════════════════════════════════════════════════════════════════
async def cmd_provider(parts: List[str]) -> bool:  # noqa: D401
    """Handle the `/provider` slash-command inside chat."""
    # Use global context manager
    context = get_context()

    # Ensure we have a model_manager
    if not context.model_manager:
        log.debug("Creating ModelManager for chat provider command")
        from mcp_cli.model_manager import ModelManager

        context.model_manager = ModelManager()

    # Store current provider/model for comparison
    old_provider = context.provider
    old_model = context.model

    try:
        # Forward everything after the command itself to the shared helper
        # Note: provider_action_async may still expect a dict, so pass one
        provider_ctx = {
            "model_manager": context.model_manager,
            "provider": context.provider,
            "model": context.model,
        }
        await provider_action_async(parts[1:])

        # Update global context with any changes
        if "provider" in provider_ctx:
            context.provider = provider_ctx["provider"]
        if "model" in provider_ctx:
            context.model = provider_ctx["model"]

        # Check if provider/model changed and provide chat-specific feedback
        new_provider = context.provider
        new_model = context.model

        if (new_provider != old_provider or new_model != old_model) and new_provider:
            output.success(f"Chat session now using: {new_provider}/{new_model}")
            output.print("Future messages will use the new provider.")

    except Exception as exc:  # pragma: no cover – unexpected edge cases
        output.error(f"Provider command failed: {exc}")
        log.exception("Chat provider command error")

        # Provide chat-specific troubleshooting hints
        if "available_models" in str(exc) or "models" in str(exc):
            output.warning("Chat troubleshooting:")
            output.print("  • This might be a chuk-llm 0.7 compatibility issue")
            output.print("  • Try: /provider list to see current provider status")
            output.print(
                f"  • Current context: provider={context.provider}, model={context.model}"
            )

    return True


# ════════════════════════════════════════════════════════════════════════════
# /providers entry-point (plural - defaults to list)
# ════════════════════════════════════════════════════════════════════════════
async def cmd_providers(parts: List[str]) -> bool:  # noqa: D401
    """Handle the `/providers` slash-command inside chat (defaults to list)."""
    # Use global context manager
    context = get_context()

    # Ensure we have a model_manager
    if not context.model_manager:
        log.debug("Creating ModelManager for chat providers command")
        from mcp_cli.model_manager import ModelManager

        context.model_manager = ModelManager()

    try:
        # If no subcommand provided, default to "list"
        if len(parts) <= 1:
            args = ["list"]
        else:
            # Forward the rest of the arguments
            args = parts[1:]

        # Forward to the shared helper
        # Note: provider_action_async may still expect a dict, so pass one
        # Context is already initialized in the global context manager
        await provider_action_async(args)

    except Exception as exc:  # pragma: no cover – unexpected edge cases
        output.error(f"Providers command failed: {exc}")
        log.exception("Chat providers command error")

    return True


# Additional chat-specific helper command
async def cmd_model(parts: List[str]) -> bool:
    """Quick model switcher for chat - `/model <model_name>`"""
    # Use global context manager
    context = get_context()

    if len(parts) < 2:
        # Show current model
        current_provider = context.provider
        current_model = context.model
        output.info(f"Current model: {current_provider}/{current_model}")

        # Show available models for current provider
        try:
            from mcp_cli.model_manager import ModelManager

            mm = ModelManager()
            models = mm.get_available_models(current_provider)
            if models:
                output.info(f"Available models for {current_provider}:")
                index = 0
                for model in models:  # Show first 10
                    if index == 10:
                        output.print(f"  ... and {len(models) - index} more")
                        # Use chuk_term confirm prompt
                        if not confirm(
                            "Do you want to list more models?", default=True
                        ):
                            break
                    marker = "→ " if model == current_model else "   "
                    output.print(f"  {marker}{model}")
                    index += 1
            else:
                output.info(f"No models found for provider {current_provider}")
        except Exception as e:
            output.warning(f"Could not list models: {e}")

        return True

    # Switch to specific model
    model_name = parts[1]
    current_provider = context.provider

    try:
        # Use the provider command to switch model
        # Note: provider_action_async may still expect a dict, so pass one
        # The provider_action_async function gets context internally
        await provider_action_async([current_provider, model_name])

    except Exception as exc:
        output.error(f"Model switch failed: {exc}")
        output.hint(f"Try: /provider {current_provider} {model_name}")

    return True


# ────────────────────────────────────────────────────────────────────────────
# registration
# ────────────────────────────────────────────────────────────────────────────
register_command("/provider", cmd_provider)
register_command("/providers", cmd_providers)  # NEW: Plural support
register_command("/model", cmd_model)  # Convenient shortcut for model switching
