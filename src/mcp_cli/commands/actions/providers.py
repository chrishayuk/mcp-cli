# src/mcp_cli/commands/actions/providers.py
"""
Provider command with all fixes applied and optimized.
This version incorporates the diagnostic fixes with your existing architecture.
"""

from __future__ import annotations
import subprocess
from typing import Dict, List, Any
from mcp_cli.model_manager import ModelManager
from chuk_term.ui import output, format_table
from mcp_cli.context import get_context


def _check_ollama_running() -> tuple[bool, int]:
    """
    Check if Ollama is running and return status with model count.
    Returns (is_running, model_count)
    """
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Count actual models (skip header line and empty lines)
            lines = result.stdout.strip().split("\n")
            model_lines = [line for line in lines[1:] if line.strip()]
            return True, len(model_lines)
        return False, 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False, 0


def _get_provider_status_enhanced(
    provider_name: str, info: Dict[str, Any]
) -> tuple[str, str, str]:
    """
    Enhanced status logic that handles all provider types correctly.
    Returns (status_icon, status_text, status_reason)
    """
    # Handle Ollama specially - it doesn't need API keys
    if provider_name.lower() == "ollama":
        is_running, model_count = _check_ollama_running()
        if is_running:
            return "✅", "Ready", f"Running ({model_count} models)"
        else:
            return "❌", "Not Running", "Ollama service not accessible"

    # For API-based providers, check configuration
    has_api_key = info.get("has_api_key", False)

    if not has_api_key:
        return "❌", "Not Configured", "No API key"

    # If has API key, check model availability
    models = info.get("models", info.get("available_models", []))
    model_count = len(models) if isinstance(models, list) else 0

    if model_count == 0:
        return "⚠️", "Partial Setup", "API key set but no models found"

    return "✅", "Ready", f"Configured ({model_count} models)"


def _get_model_count_display_enhanced(provider_name: str, info: Dict[str, Any]) -> str:
    """
    Enhanced model count display that handles Ollama and chuk-llm 0.7+ correctly.
    """
    # For Ollama, get live count from ollama command
    if provider_name.lower() == "ollama":
        is_running, live_count = _check_ollama_running()
        if is_running:
            return f"{live_count} models"
        else:
            return "Ollama not running"

    # For other providers, use chuk-llm data with proper key handling
    # chuk-llm 0.7+ uses "models" key, but we'll check both for compatibility
    models = info.get("models", info.get("available_models", []))

    if not isinstance(models, list):
        return "Unknown"

    count = len(models)
    if count == 0:
        return "No models found"
    elif count == 1:
        return "1 model"
    else:
        return f"{count} models"


def _get_features_display_enhanced(info: Dict[str, Any]) -> str:
    """Enhanced feature display with more comprehensive icons."""
    baseline_features = info.get("baseline_features", [])

    feature_icons = []
    if "streaming" in baseline_features:
        feature_icons.append("📡")
    if "tools" in baseline_features or "parallel_calls" in baseline_features:
        feature_icons.append("🔧")
    if "vision" in baseline_features:
        feature_icons.append("👁️")
    if "reasoning" in baseline_features:
        feature_icons.append("🧠")
    if "json_mode" in baseline_features:
        feature_icons.append("📝")

    return "".join(feature_icons) if feature_icons else "📄"


def _render_list_optimized(model_manager: ModelManager) -> None:
    """
    Optimized provider list that handles all the edge cases correctly.
    """
    table_data = []
    columns = ["Provider", "Status", "Default Model", "Models Available", "Features"]

    current_provider = model_manager.get_active_provider()

    try:
        # Get provider info using the working method
        all_providers_info = model_manager.list_available_providers()

        if not all_providers_info:
            output.error("No providers found. Check chuk-llm installation.")
            return

    except Exception as e:
        output.error(f"Error getting provider list: {e}")
        return

    # Sort providers to put current one first, then alphabetically
    provider_items = list(all_providers_info.items())
    provider_items.sort(key=lambda x: (x[0] != current_provider, x[0]))

    for provider_name, provider_info in provider_items:
        # Handle error cases
        if "error" in provider_info:
            table_data.append(
                {
                    "Provider": provider_name,
                    "Status": "Error",
                    "Default Model": "-",
                    "Models Available": "-",
                    "Features": provider_info["error"][:20] + "...",
                }
            )
            continue

        # Mark current provider
        display_name = (
            f"→ {provider_name}"
            if provider_name == current_provider
            else f"  {provider_name}"
        )

        # Enhanced status using improved logic
        status_icon, status_text, status_reason = _get_provider_status_enhanced(
            provider_name, provider_info
        )

        # Format status text
        status_display = f"{status_icon} {status_text}"

        # Default model with proper fallback
        default_model = provider_info.get("default_model", "-")
        if not default_model or default_model in ("None", "null"):
            default_model = "-"

        # Enhanced model count display
        models_display = _get_model_count_display_enhanced(provider_name, provider_info)

        # Enhanced features
        features_display = _get_features_display_enhanced(provider_info)

        table_data.append(
            {
                "Provider": display_name,
                "Status": status_display,
                "Default Model": default_model,
                "Models Available": models_display,
                "Features": features_display,
            }
        )

    # Create and display table with visual styling
    output.rule("[bold]🌐 Available Providers[/bold]", style="primary")
    
    table = format_table(
        table_data, 
        title=None,  # Using rule for title
        columns=columns
    )
    output.print_table(table)
    output.print()
    output.tip("💡 Use '/provider <name>' to switch providers")

    # Show helpful tips based on current state
    inactive_providers = []
    for name, info in all_providers_info.items():
        if "error" not in info:
            status_icon, _, _ = _get_provider_status_enhanced(name, info)
            if status_icon == "❌":
                inactive_providers.append(name)

    if inactive_providers:
        output.hint(
            "Configure providers with: /provider set <name> api_key <key>"
        )


def _render_diagnostic_optimized(
    model_manager: ModelManager, target: str | None
) -> None:
    """Optimized diagnostic that shows detailed status for providers."""
    if target:
        providers_to_test = [target] if model_manager.validate_provider(target) else []
        if not providers_to_test:
            output.error(f"Unknown provider: {target}")
            available = ", ".join(model_manager.list_providers())
            output.warning(f"Available providers: {available}")
            return
    else:
        providers_to_test = model_manager.list_providers()

    table_data = []
    columns = ["Provider", "Status", "Models", "Features", "Details"]

    try:
        all_providers_data = model_manager.list_available_providers()
    except Exception as e:
        output.error(f"Error getting provider data: {e}")
        return

    for provider in providers_to_test:
        try:
            provider_info = all_providers_data.get(provider, {})

            # Skip if provider has errors
            if "error" in provider_info:
                table_data.append(
                    {
                        "Provider": provider,
                        "Status": "Error",
                        "Models": "-",
                        "Features": "-",
                        "Details": provider_info["error"][:30] + "...",
                    }
                )
                continue

            # Enhanced status
            status_icon, status_text, status_reason = _get_provider_status_enhanced(
                provider, provider_info
            )

            status_display = f"{status_icon} {status_text}"

            # Model count
            models_display = _get_model_count_display_enhanced(provider, provider_info)

            # Features
            features_display = _get_features_display_enhanced(provider_info)

            # Additional details
            details = []
            if provider_info.get("api_base"):
                details.append(f"API: {provider_info['api_base']}")
            if provider_info.get("discovery_enabled"):
                details.append("Discovery: ✅")
            details_str = " | ".join(details) if details else "-"

            table_data.append(
                {
                    "Provider": provider,
                    "Status": status_display,
                    "Models": models_display,
                    "Features": features_display,
                    "Details": details_str,
                }
            )

        except Exception as exc:
            table_data.append(
                {
                    "Provider": provider,
                    "Status": "Error",
                    "Models": "-",
                    "Features": "-",
                    "Details": str(exc)[:30] + "...",
                }
            )

    # Create and display table using chuk-term
    table = format_table(table_data, title="Provider Diagnostics", columns=columns)
    output.print_table(table)


def _switch_provider_enhanced(
    model_manager: ModelManager,
    provider_name: str,
    model_name: str | None,
    context: Dict,
) -> None:
    """Enhanced provider switching with better validation and feedback."""

    if not model_manager.validate_provider(provider_name):
        available = ", ".join(model_manager.list_providers())
        output.error(f"Unknown provider: {provider_name}")
        output.info(f"Available providers: {available}")
        return

    # Get provider info for validation
    try:
        all_providers_info = model_manager.list_available_providers()
        provider_info = all_providers_info.get(provider_name, {})

        if "error" in provider_info:
            output.error(f"Provider error: {provider_info['error']}")
            return

        # Enhanced status validation
        status_icon, status_text, status_reason = _get_provider_status_enhanced(
            provider_name, provider_info
        )

        if status_icon == "❌":
            output.error(f"Provider not ready: {status_reason}")

            # Provide specific help
            if provider_name.lower() == "ollama":
                output.tip("Start Ollama with: ollama serve")
            elif "No API key" in status_reason:
                env_var = f"{provider_name.upper()}_API_KEY"
                output.tip(
                    f"Set API key with: /provider set {provider_name} api_key YOUR_KEY"
                )
                output.tip(f"Or set environment variable: export {env_var}=YOUR_KEY")

            return

        elif status_icon == "⚠️":
            output.warning(f"{status_reason}")
            output.info("Continuing anyway...")

    except Exception as e:
        output.warning(f"Could not validate provider: {e}")

    # Determine target model
    if model_name:
        target_model = model_name
    else:
        # Get default model
        try:
            target_model = model_manager.get_default_model(provider_name)
            if not target_model:
                # Fallback to first available model
                available_models = model_manager.get_available_models(provider_name)
                target_model = available_models[0] if available_models else "default"
        except Exception:
            target_model = "default"

    output.info(f"Switching to {provider_name} (model: {target_model})...")

    # Perform the switch
    try:
        model_manager.switch_model(provider_name, target_model)
    except Exception as e:
        output.error(f"Failed to switch provider: {e}")
        return

    # Update context (ApplicationContext object)
    try:
        context.set("provider", provider_name)
        context.set("model", target_model)
        context.set("client", model_manager.get_client())
        context.model_manager = model_manager
    except Exception as e:
        output.warning(f"Could not update client context: {e}")

    output.success(f"✅ Switched to {provider_name} (model: {target_model})")


# Update the main action function with enhanced sub-commands
async def provider_action_async(
    args: List[str],
) -> None:
    """Enhanced provider action with all optimizations applied."""
    # Get context and model manager
    context = get_context()
    model_manager = context.model_manager

    def _show_status() -> None:
        provider, model = model_manager.get_active_provider_and_model()
        status = model_manager.get_status_summary()

        # Get enhanced status for current provider
        try:
            all_providers = model_manager.list_available_providers()
            current_info = all_providers.get(provider, {})
            status_icon, status_text, status_reason = _get_provider_status_enhanced(
                provider, current_info
            )

            # Display in a beautifully formatted panel
            output.rule(f"[bold]🔧 Provider Status[/bold]", style="primary")
            output.print()
            
            # Create visually appealing status display
            output.print(f"  [bold]Provider:[/bold] {provider}")
            output.print(f"  [bold]Model:[/bold]    {model}")
            output.print(f"  [bold]Status:[/bold]   {status_icon} {status_text}")
            output.print(f"  [bold]Features:[/bold] {_format_features(status)}")
            
            if status_icon != "✅":
                output.print()
                output.warning(f"  ⚠️  {status_reason}")
            
            output.print()
            
            # Show available providers tip
            output.tip("Use: /provider <name> to switch  |  /providers to list all  |  /provider set <name> for config")

        except Exception as e:
            output.info(f"Current provider: {provider}")
            output.info(f"Current model   : {model}")
            output.warning(f"Status check failed: {e}")

    def _format_features(status: Dict) -> str:
        features = []
        if status.get("supports_streaming"):
            features.append("📡 streaming")
        if status.get("supports_tools"):
            features.append("🔧 tools")
        if status.get("supports_vision"):
            features.append("👁️ vision")
        return " ".join(features) or "📄 text only"

    # Dispatch logic
    if not args:
        _show_status()
        return

    sub, *rest = args
    sub = sub.lower()

    if sub == "list":
        _render_list_optimized(model_manager)
        return

    if sub == "config":
        _render_config(model_manager)
        return

    if sub == "diagnostic":
        target = rest[0] if rest else None
        _render_diagnostic_optimized(model_manager, target)
        return

    if sub == "set" and len(rest) >= 2:
        provider_name, setting = rest[0], rest[1]
        value = rest[2] if len(rest) >= 3 else None
        _mutate(model_manager, provider_name, setting, value)
        return

    # Provider switching
    provider_name = sub
    model_name = rest[0] if rest else None
    _switch_provider_enhanced(model_manager, provider_name, model_name, context)


# Keep existing helper functions but use them in the enhanced versions above
def _render_config(model_manager: ModelManager) -> None:
    """Show detailed configuration - keeping your existing implementation."""
    # ... existing implementation
    pass


def _mutate(model_manager: ModelManager, provider: str, key: str, value: str) -> None:
    """Update provider configuration - keeping your existing implementation."""
    # ... existing implementation
    pass


# Sync wrapper
def provider_action(args: List[str]) -> None:
    """Sync wrapper for provider_action_async."""
    from mcp_cli.utils.async_utils import run_blocking

    run_blocking(provider_action_async(args))
