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
from typing import Any, Dict, List

# Cross-platform Rich console helper
from chuk_term.ui import output
from rich.prompt import Prompt

# Shared implementation
from mcp_cli.commands.provider import provider_action_async
from mcp_cli.chat.commands import register_command

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# /provider entry-point
# ════════════════════════════════════════════════════════════════════════════
async def cmd_provider(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Handle the `/provider` slash-command inside chat."""

    # Ensure we have a model_manager in the chat context
    if "model_manager" not in ctx:
        log.debug("Creating ModelManager for chat provider command")
        from mcp_cli.model_manager import ModelManager

        ctx["model_manager"] = ModelManager()

    # Store current provider/model for comparison
    old_provider = ctx.get("provider")
    old_model = ctx.get("model")

    try:
        # Forward everything after the command itself to the shared helper
        await provider_action_async(parts[1:], context=ctx)

        # Check if provider/model changed and provide chat-specific feedback
        new_provider = ctx.get("provider")
        new_model = ctx.get("model")

        if (new_provider != old_provider or new_model != old_model) and new_provider:
            output.print(
                f"[green]Chat session now using:[/green] {new_provider}/{new_model}"
            )
            output.print("[dim]Future messages will use the new provider.[/dim]")

    except Exception as exc:  # pragma: no cover – unexpected edge cases
        output.print(f"[red]Provider command failed:[/red] {exc}")
        log.exception("Chat provider command error")

        # Provide chat-specific troubleshooting hints
        if "available_models" in str(exc) or "models" in str(exc):
            output.print("[yellow]Chat troubleshooting:[/yellow]")
            output.print("  • This might be a chuk-llm 0.7 compatibility issue")
            output.print("  • Try: /provider list to see current provider status")
            output.print(
                f"  • Current context: provider={ctx.get('provider')}, model={ctx.get('model')}"
            )

    return True


# ════════════════════════════════════════════════════════════════════════════
# /providers entry-point (plural - defaults to list)
# ════════════════════════════════════════════════════════════════════════════
async def cmd_providers(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Handle the `/providers` slash-command inside chat (defaults to list)."""

    # Ensure we have a model_manager in the chat context
    if "model_manager" not in ctx:
        log.debug("Creating ModelManager for chat providers command")
        from mcp_cli.model_manager import ModelManager

        ctx["model_manager"] = ModelManager()

    try:
        # If no subcommand provided, default to "list"
        if len(parts) <= 1:
            args = ["list"]
        else:
            # Forward the rest of the arguments
            args = parts[1:]

        # Forward to the shared helper
        await provider_action_async(args, context=ctx)

    except Exception as exc:  # pragma: no cover – unexpected edge cases
        output.print(f"[red]Providers command failed:[/red] {exc}")
        log.exception("Chat providers command error")

    return True


# Additional chat-specific helper command
async def cmd_model(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Quick model switcher for chat - `/model <model_name>`"""

    if len(parts) < 2:
        # Show current model
        current_provider = ctx.get("provider", "unknown")
        current_model = ctx.get("model", "unknown")
        output.print(f"[cyan]Current model:[/cyan] {current_provider}/{current_model}")

        # Show available models for current provider
        try:
            from mcp_cli.model_manager import ModelManager

            mm = ModelManager()
            models = mm.get_available_models(current_provider)
            if models:
                output.print(f"[cyan]Available models for {current_provider}:[/cyan]")
                index = 0
                for model in models:  # Show first 10
                    if index == 10:
                        output.print(f"  ... and {len(models) - index} more")
                        # Use rich to display the prompt, fallback to input() if needed
                        prompt_text = (
                            "[bold white]Do you want to list more models?[/bold white]"
                        )
                        try:
                            # Use rich Prompt if available
                            response = Prompt.ask(
                                prompt_text,
                                case_sensitive=False,
                                choices=["y", "n"],
                                default="y",
                            )
                            response = response.strip().lower()
                        except Exception:
                            # Fallback to input()
                            print(prompt_text, end="")
                            response = input().strip().lower()
                        if response not in ["y", ""]:
                            break
                    marker = "→ " if model == current_model else "   "
                    output.print(f"  {marker}{model}")
                    index += 1
            else:
                output.print(
                    f"[cyan]No models found for provider {current_provider}[/cyan]"
                )
        except Exception as e:
            output.print(f"[yellow]Could not list models:[/yellow] {e}")

        return True

    # Switch to specific model
    model_name = parts[1]
    current_provider = ctx.get("provider", "openai")

    try:
        # Use the provider command to switch model
        await provider_action_async([current_provider, model_name], context=ctx)
    except Exception as exc:
        output.print(f"[red]Model switch failed:[/red] {exc}")
        output.print(f"[yellow]Try:[/yellow] /provider {current_provider} {model_name}")

    return True


# ────────────────────────────────────────────────────────────────────────────
# registration
# ────────────────────────────────────────────────────────────────────────────
register_command("/provider", cmd_provider)
register_command("/providers", cmd_providers)  # NEW: Plural support
register_command("/model", cmd_model)  # Convenient shortcut for model switching
