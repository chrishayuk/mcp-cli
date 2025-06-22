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

Features
--------
* **Cross-platform Rich console** - via
  :pyfunc:`mcp_cli.utils.rich_helpers.get_console`.
* **Graceful error surfacing** - unexpected exceptions are caught and printed
  as red error messages instead of exploding the event-loop.
* **Chat context awareness** - preserves chat session state when switching providers.
* **Plural support** - `/providers` defaults to listing all providers.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List

# Cross-platform Rich console helper
from mcp_cli.utils.rich_helpers import get_console

# Shared implementation
from mcp_cli.commands.provider import provider_action_async
from mcp_cli.chat.commands import register_command

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# /provider entry-point
# ════════════════════════════════════════════════════════════════════════════
async def cmd_provider(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Handle the `/provider` slash-command inside chat."""
    console = get_console()

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
            console.print(f"[green]Chat session now using:[/green] {new_provider}/{new_model}")
            console.print(f"[dim]Future messages will use the new provider.[/dim]")
            
    except Exception as exc:  # pragma: no cover – unexpected edge cases
        console.print(f"[red]Provider command failed:[/red] {exc}")
        log.exception("Chat provider command error")
        
        # Provide chat-specific troubleshooting hints
        if "available_models" in str(exc) or "models" in str(exc):
            console.print(f"[yellow]Chat troubleshooting:[/yellow]")
            console.print(f"  • This might be a chuk-llm 0.7 compatibility issue")
            console.print(f"  • Try: /provider list to see current provider status")
            console.print(f"  • Current context: provider={ctx.get('provider')}, model={ctx.get('model')}")

    return True


# ════════════════════════════════════════════════════════════════════════════
# /providers entry-point (plural - defaults to list)
# ════════════════════════════════════════════════════════════════════════════
async def cmd_providers(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Handle the `/providers` slash-command inside chat (defaults to list)."""
    console = get_console()

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
        console.print(f"[red]Providers command failed:[/red] {exc}")
        log.exception("Chat providers command error")

    return True


# Additional chat-specific helper command
async def cmd_model(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Quick model switcher for chat - `/model <model_name>`"""
    console = get_console()
    
    if len(parts) < 2:
        # Show current model
        current_provider = ctx.get("provider", "unknown")
        current_model = ctx.get("model", "unknown")
        console.print(f"[cyan]Current model:[/cyan] {current_provider}/{current_model}")
        
        # Show available models for current provider
        try:
            from mcp_cli.model_manager import ModelManager
            mm = ModelManager()
            models = mm.get_available_models(current_provider)
            if models:
                console.print(f"[cyan]Available models for {current_provider}:[/cyan]")
                for model in models[:10]:  # Show first 10
                    marker = "→ " if model == current_model else "   "
                    console.print(f"  {marker}{model}")
                if len(models) > 10:
                    console.print(f"  ... and {len(models) - 10} more")
        except Exception as e:
            console.print(f"[yellow]Could not list models:[/yellow] {e}")
        
        return True
    
    # Switch to specific model
    model_name = parts[1]
    current_provider = ctx.get("provider", "openai")
    
    try:
        # Use the provider command to switch model
        await provider_action_async([current_provider, model_name], context=ctx)
    except Exception as exc:
        console.print(f"[red]Model switch failed:[/red] {exc}")
        console.print(f"[yellow]Try:[/yellow] /provider {current_provider} {model_name}")
    
    return True


# ────────────────────────────────────────────────────────────────────────────
# registration
# ────────────────────────────────────────────────────────────────────────────
register_command("/provider", cmd_provider)
register_command("/providers", cmd_providers)  # NEW: Plural support
register_command("/model", cmd_model)  # Convenient shortcut for model switching