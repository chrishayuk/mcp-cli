# mcp_cli/commands/model.py
"""
Model management command for MCP-CLI.

Commands:
  /model                → show current model & provider
  /model list           → list all available models
  /model <name>         → switch to a different model
  /model refresh        → refresh model discovery
"""

from __future__ import annotations

from typing import Any, Dict, List

from chuk_term.ui import output, format_table
from mcp_cli.model_manager import ModelManager
from mcp_cli.utils.async_utils import run_blocking
from mcp_cli.utils.llm_probe import LLMProbe


async def model_action_async(args: List[str], *, context: Dict[str, Any]) -> None:
    """
    Handle model management commands.

    Args:
        args: Command arguments
        context: Execution context containing model_manager
    """
    # Get or create ModelManager
    model_manager: ModelManager = context.get("model_manager") or ModelManager()
    context.setdefault("model_manager", model_manager)

    provider = model_manager.get_active_provider()
    current_model = model_manager.get_active_model()

    # No arguments - show current status
    if not args:
        await _show_status(model_manager, current_model, provider)
        return

    command = args[0].lower()

    # Handle subcommands
    if command == "list":
        await _list_models(model_manager, provider, current_model)
    elif command == "refresh":
        await _refresh_models(model_manager, provider)
    else:
        # Assume it's a model name to switch to
        await _switch_model(args[0], model_manager, provider, current_model, context)


async def _show_status(model_manager: ModelManager, model: str, provider: str) -> None:
    """Show current model status."""
    output.info(f"Current model: {provider}/{model}")

    # Get available models
    available_models = model_manager.get_available_models(provider)

    if not available_models:
        output.warning("No models found for current provider")
        return

    # Show first few available models
    output.print("\nAvailable models:")
    count = 0
    for available_model in available_models:
        if available_model == model:
            output.print(f"  → [bold green]{available_model}[/bold green] (current)")
        else:
            output.print(f"     {available_model}")

        count += 1
        if count >= 10:
            remaining = len(available_models) - 10
            if remaining > 0:
                output.print(f"     ... and {remaining} more")
            break

    # Show Ollama status if applicable
    if provider.lower() == "ollama":
        await _show_ollama_status(model_manager)

    output.hint(
        "\nUse: /model <name> to switch  |  /model list for all  |  /model refresh to discover"
    )


async def _list_models(
    model_manager: ModelManager, provider: str, current_model: str
) -> None:
    """List all available models."""
    available_models = model_manager.get_available_models(provider)

    if not available_models:
        output.error(f"No models found for provider '{provider}'")
        return

    # Build table data
    table_data = []

    # Get local Ollama models if applicable
    local_models = []
    if provider.lower() == "ollama":
        ollama_running, local_models = await _check_local_ollama()

    # Get static models from config
    static_models = set()
    try:
        provider_info = model_manager.get_provider_info(provider)
        static_models = set(provider_info.get("models", []))
    except Exception:
        pass

    # Build rows
    for model_name in available_models:
        # Determine status and type
        if model_name == current_model:
            status = "→ Current"
        else:
            status = ""

        if model_name in static_models:
            model_type = "Static"
        elif model_name in local_models:
            model_type = "Local"
        else:
            model_type = "Discovered"

        # Add info
        info = []
        if ":latest" in model_name:
            info.append("latest")
        if "embed" in model_name.lower():
            info.append("embedding")

        table_data.append(
            {
                "Status": status,
                "Model": model_name,
                "Type": model_type,
                "Info": ", ".join(info) if info else "-",
            }
        )

    # Display table
    table = format_table(
        table_data,
        title=f"Models for {provider} ({len(available_models)} total)",
        columns=["Status", "Model", "Type", "Info"],
    )
    output.print_table(table)

    output.tip("Use '/model <name>' to switch to any model")


async def _refresh_models(model_manager: ModelManager, provider: str) -> None:
    """Refresh model discovery."""
    with output.loading(f"Refreshing models for {provider}..."):
        before_count = len(model_manager.get_available_models(provider))

        try:
            success = model_manager.refresh_discovery(provider)

            if success:
                after_count = len(model_manager.get_available_models(provider))
                new_count = after_count - before_count

                if new_count > 0:
                    output.success(f"Discovered {new_count} new models!")
                else:
                    output.info("No new models discovered")

                output.print(f"Total models: {after_count}")
            else:
                output.error("Refresh failed")

        except Exception as e:
            output.error(f"Refresh error: {e}")


async def _switch_model(
    new_model: str,
    model_manager: ModelManager,
    provider: str,
    current_model: str,
    context: Dict[str, Any],
) -> None:
    """Attempt to switch to a new model."""
    with output.loading(f"Testing model '{new_model}'..."):
        try:
            # Validate model
            is_valid = model_manager.validate_model_for_provider(provider, new_model)

            if not is_valid:
                output.error(f"Model not available: {new_model}")

                # Show suggestions
                available = model_manager.get_available_models(provider)
                if available:
                    suggestions = available[:5]
                    output.tip(f"Available models: {', '.join(suggestions)}")
                    if len(available) > 5:
                        output.print(f"... and {len(available) - 5} more")
                return

            # Test the model
            async with LLMProbe(model_manager, suppress_logging=True) as probe:
                result = await probe.test_model(new_model)

            if result.success:
                # Switch successful
                model_manager.set_active_model(new_model)
                context["model"] = new_model
                context["client"] = result.client
                context["model_manager"] = model_manager
                output.success(f"Switched to model: {new_model}")
            else:
                error_msg = result.error_message or "Model test failed"
                output.error(f"Model test failed: {error_msg}")
                output.warning(f"Keeping current model: {current_model}")

        except Exception as e:
            output.error(f"Model switch failed: {e}")
            output.warning(f"Keeping current model: {current_model}")


async def _show_ollama_status(model_manager: ModelManager) -> None:
    """Show Ollama-specific status information."""
    try:
        ollama_running, local_models = await _check_local_ollama()

        if ollama_running:
            available = len(model_manager.get_available_models("ollama"))
            discovery = model_manager.get_discovery_status()
            enabled = discovery.get("ollama_enabled", False)

            status = f"Ollama: {len(local_models)} local, {available} accessible"
            if enabled:
                status += " | Discovery: ✅"
            else:
                status += " | Discovery: ❌"

            output.print(f"\n[dim]{status}[/dim]")
        else:
            output.print(
                "\n[dim]Ollama: Not running | Use 'ollama serve' to start[/dim]"
            )

    except Exception:
        pass


async def _check_local_ollama() -> tuple[bool, list[str]]:
    """Check if Ollama is running and get local models."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            data = response.json()

            models = [m["name"] for m in data.get("models", [])]
            return True, models

    except Exception:
        return False, []


def model_action(args: List[str], *, context: Dict[str, Any]) -> None:
    """Synchronous wrapper for model_action_async."""
    run_blocking(model_action_async(args, context=context))
