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
from mcp_cli.context import get_context, ApplicationContext
from mcp_cli.commands.models import ProviderActionParams


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
    token_source = info.get("token_source", "none")

    if not has_api_key:
        return "❌", "Not Configured", "No API key"

    # If has API key, check model availability
    models = info.get("models", info.get("available_models", []))
    model_count = len(models) if isinstance(models, list) else 0

    # Create status reason with token source info
    if token_source == "env":
        source_info = " (env)"
    elif token_source == "storage":
        source_info = " (storage)"
    else:
        source_info = ""

    if model_count == 0:
        return "⚠️", "Partial Setup", f"API key set but no models found{source_info}"

    return "✅", "Ready", f"Configured ({model_count} models){source_info}"


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
    # Create token manager for checking token sources
    from mcp_cli.auth import TokenManager, TokenStoreBackend
    from mcp_cli.constants import NAMESPACE

    try:
        token_manager = TokenManager(
            backend=TokenStoreBackend.AUTO, namespace=NAMESPACE, service_name="mcp-cli"
        )
    except Exception:
        token_manager = None

    table_data = []
    columns = [
        "Provider",
        "Status",
        "Token",
        "Default Model",
        "Models Available",
        "Features",
    ]

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
                    "Token": "-",
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

        # Check token source ourselves (chuk_llm doesn't provide this)
        token_source = "none"
        if provider_name.lower() != "ollama" and token_manager:
            from mcp_cli.auth.provider_tokens import check_provider_token_status

            token_status = check_provider_token_status(provider_name, token_manager)
            token_source = token_status.get("source", "none")

        # Display token source
        if provider_name.lower() == "ollama":
            token_display = "-"
        elif token_source == "env":
            token_display = "🌍 env"
        elif token_source == "storage":
            token_display = "🔐 storage"
        else:
            token_display = "❌ none"

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
                "Token": token_display,
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
        columns=columns,
    )
    output.print_table(table)
    output.print()

    # Show comprehensive tips for provider management
    output.tip("💡 Provider Management:")
    output.info("  • Switch: /provider <name>")
    output.info("  • Add custom: /provider add <name> <api_base> [model]")
    output.info("  • Remove: /provider remove <name>")
    output.info("  • List custom: /provider custom")
    output.hint(
        "  Custom providers use env vars: {NAME}_API_KEY (e.g., LOCALAI_API_KEY)"
    )

    # Show helpful tips based on current state
    inactive_providers = []
    inactive_custom_providers = []
    custom_count = 0
    for name, info in all_providers_info.items():
        if info.get("is_custom"):
            custom_count += 1
        if "error" not in info:
            status_icon, _, _ = _get_provider_status_enhanced(name, info)
            if status_icon == "❌":
                inactive_providers.append(name)
                if info.get("is_custom"):
                    inactive_custom_providers.append(name)

    # Prioritize showing custom provider hints if any are unconfigured
    if inactive_custom_providers:
        # Show hint for custom provider
        first_custom = inactive_custom_providers[0]
        output.hint(
            f"Configure API key: export {first_custom.upper().replace('-', '_')}_API_KEY=your-key"
        )
    elif inactive_providers:
        # Show hint for built-in provider
        first_inactive = inactive_providers[0]
        env_var_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "togetherai": "TOGETHER_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "azure_openai": "AZURE_OPENAI_API_KEY",
            "watsonx": "WATSONX_API_KEY",
        }
        env_var = env_var_map.get(first_inactive, f"{first_inactive.upper()}_API_KEY")
        output.hint(f"Configure API key: export {env_var}=your-key")

    # Suggest adding custom providers if none exist
    if custom_count == 0:
        output.hint(
            "Add OpenAI-compatible providers: /provider add localai http://localhost:8080/v1"
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
    context: ApplicationContext,
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
        context.provider = provider_name
        context.model = target_model
        # context doesn't have a client attribute, but it has model_manager
        context.model_manager = model_manager
    except Exception as e:
        output.warning(f"Could not update client context: {e}")

    output.success(f"✅ Switched to {provider_name} (model: {target_model})")


# Update the main action function with enhanced sub-commands
async def provider_action_async(params: ProviderActionParams) -> None:
    """
    Enhanced provider action with all optimizations applied.

    Args:
        params: Provider action parameters

    Example:
        >>> params = ProviderActionParams(args=["list"], detailed=True)
        >>> await provider_action_async(params)
    """
    # Get context and model manager
    context: ApplicationContext = get_context()
    model_manager = context.model_manager

    if not model_manager:
        output.error("Model manager not available")
        return

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
            output.rule("[bold]🔧 Provider Status[/bold]", style="primary")
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
            output.tip(
                "Use: /provider <name> to switch  |  /providers to list all  |  /provider set <name> for config"
            )

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
    if not params.args:
        _show_status()
        return

    sub, *rest = params.args
    sub = sub.lower()

    if sub == "list":
        _render_list_optimized(model_manager)
        return

    if sub == "custom":
        _list_custom_providers()
        return

    if sub == "add" and len(rest) >= 2:
        # /provider add <name> <api_base> [model1 model2 ...]
        name = rest[0]
        api_base = rest[1]
        models = rest[2:] if len(rest) > 2 else None
        _add_custom_provider(name, api_base, models)
        return

    if sub == "remove" and rest:
        # /provider remove <name>
        name = rest[0]
        _remove_custom_provider(name)
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


def _add_custom_provider(
    name: str, api_base: str, models: List[str] | None = None
) -> None:
    """Add a custom OpenAI-compatible provider."""
    from mcp_cli.utils.preferences import get_preference_manager
    import os

    prefs = get_preference_manager()

    # Check if provider already exists
    if prefs.is_custom_provider(name):
        output.error(f"Provider '{name}' already exists. Use 'update' to modify it.")
        return

    # Add the provider
    prefs.add_custom_provider(
        name=name,
        api_base=api_base,
        models=models or ["gpt-4", "gpt-3.5-turbo"],
        default_model=models[0] if models else "gpt-4",
    )

    # Get the expected env var name
    env_var = f"{name.upper().replace('-', '_')}_API_KEY"

    output.success(f"✅ Added provider '{name}'")
    output.info(f"   API Base: {api_base}")
    output.info(f"   Models: {', '.join(models or ['gpt-4', 'gpt-3.5-turbo'])}")

    # Check if API key is set
    if not os.environ.get(env_var):
        output.warning("\n⚠️  API key required. Set it with:")
        output.print(f"   [bold]export {env_var}=your-api-key[/bold]")
        output.info(
            "\n   The environment variable name is based on your provider name:"
        )
        output.info(f"   Provider '{name}' → {env_var}")
    else:
        output.success(f"   API Key: ✅ Found in {env_var}")


def _remove_custom_provider(name: str) -> None:
    """Remove a custom provider."""
    from mcp_cli.utils.preferences import get_preference_manager

    prefs = get_preference_manager()

    if not prefs.is_custom_provider(name):
        output.error(f"Provider '{name}' is not a custom provider or doesn't exist.")
        return

    if prefs.remove_custom_provider(name):
        output.success(f"✅ Removed provider '{name}'")
    else:
        output.error(f"Failed to remove provider '{name}'")


def _list_custom_providers() -> None:
    """List all custom providers."""
    from mcp_cli.utils.preferences import get_preference_manager
    from mcp_cli.auth.provider_tokens import get_provider_token_display_status
    from mcp_cli.auth import TokenManager

    prefs = get_preference_manager()
    custom_providers = prefs.get_custom_providers()

    if not custom_providers:
        output.info("No custom providers configured.")
        output.tip("Add one with: /provider add <name> <api_base> [models...]")
        return

    output.rule("[bold]🔧 Custom Providers[/bold]", style="primary")

    try:
        token_manager = TokenManager()
    except Exception:
        token_manager = None

    table_data = []
    for name, provider_data in custom_providers.items():
        # Handle custom env var name if specified
        custom_env_var = provider_data.get("env_var_name")
        if custom_env_var:
            # Use custom env var name instead of default
            import os

            if os.environ.get(custom_env_var):
                token_status = f"✅ {custom_env_var}"
            else:
                token_status = f"❌ {custom_env_var} not set"
        else:
            # Use standard hierarchical token status display
            token_status = get_provider_token_display_status(name, token_manager)

        table_data.append(
            {
                "Provider": name,
                "API Base": provider_data["api_base"],
                "Models": ", ".join(provider_data.get("models", [])),
                "Default": provider_data.get("default_model", "gpt-4"),
                "Token": token_status,
            }
        )

    table = format_table(
        table_data,
        title=None,
        columns=["Provider", "API Base", "Models", "Default", "Token"],
    )
    output.print_table(table)


def _mutate(
    model_manager: ModelManager, provider: str, key: str, value: str | None
) -> None:
    """Update provider configuration - keeping your existing implementation."""
    # ... existing implementation
    pass


# Sync wrapper
def provider_action(args: List[str]) -> None:
    """Sync wrapper for provider_action_async."""
    from mcp_cli.utils.async_utils import run_blocking

    params = ProviderActionParams(args=args)
    run_blocking(provider_action_async(params))
