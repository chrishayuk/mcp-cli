# mcp_cli/commands/provider.py
"""
Fixed provider command implementation for chuk-llm 0.7 migration.

Usage Examples:
    CLI Mode:
        mcp-cli provider                      # Show current provider status
        mcp-cli provider list                 # List all providers with model counts
        mcp-cli provider config               # Show detailed configuration
        mcp-cli provider diagnostic           # Run diagnostics on all providers
        mcp-cli provider diagnostic openai    # Run diagnostics on specific provider
        mcp-cli provider anthropic            # Switch to Anthropic (default model)
        mcp-cli provider openai gpt-4o        # Switch to OpenAI with specific model
        mcp-cli provider set openai api_key sk-your-key-here
        mcp-cli provider set anthropic api_base https://api.anthropic.com
        mcp-cli provider set groq default_model llama-3.3-70b-versatile

    Chat Mode:
        /provider                         # Show current provider status
        /provider list                    # See all providers with model counts
        /provider anthropic              # Switch to Anthropic
        /provider openai gpt-4o          # Switch to OpenAI with specific model
        /provider diagnostic gemini      # Check Gemini setup
        /provider set deepseek api_key sk-... # Configure API key
        /provider config                 # Show detailed configuration
"""
from __future__ import annotations
import subprocess
from typing import Dict, List, Tuple
from rich.table import Table

from mcp_cli.model_manager import ModelManager
from mcp_cli.utils.rich_helpers import get_console

console = get_console()


def _check_ollama_running() -> bool:
    """Check if Ollama is running locally."""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def _get_provider_status(provider_name: str, info: Dict) -> tuple[str, str]:
    """
    Get proper status for a provider based on its type and configuration.
    
    Returns:
        tuple[str, str]: (status_icon, status_reason)
    """
    # Handle Ollama specially - it doesn't need API keys
    if provider_name == "ollama":
        if _check_ollama_running():
            return "✅", "Ollama running"
        else:
            return "❌", "Ollama not running"
    
    # For API-based providers, check if they have API keys
    has_api_key = info.get("has_api_key", False)
    if has_api_key:
        return "✅", "API key configured"
    else:
        return "❌", "No API key"


async def provider_action_async(
    args: List[str],
    *,
    context: Dict,
) -> None:
    """Handle all provider sub-commands using chuk-llm's unified configuration."""
    model_manager: ModelManager = context.get("model_manager") or ModelManager()
    context.setdefault("model_manager", model_manager)

    def _show_status() -> None:
        provider, model = model_manager.get_active_provider_and_model()
        status = model_manager.get_status_summary()
        
        console.print(f"[cyan]Current provider:[/cyan] {provider}")
        console.print(f"[cyan]Current model   :[/cyan] {model}")
        console.print(f"[cyan]Configured      :[/cyan] {'✅' if status['provider_configured'] else '❌'}")
        console.print(f"[cyan]Features        :[/cyan] {_format_features(status)}")

    def _format_features(status: Dict) -> str:
        features = []
        if status.get('supports_streaming'):
            features.append("📡 streaming")
        if status.get('supports_tools'):
            features.append("🔧 tools")
        if status.get('supports_vision'):
            features.append("👁️ vision")
        return " ".join(features) or "text only"

    # ── dispatch ────────────────────────────────────────────────────────
    if not args:
        _show_status()
        return

    sub, *rest = args
    sub = sub.lower()

    if sub == "list":
        _render_list(model_manager)
        return

    if sub == "config":
        _render_config(model_manager)
        return

    if sub == "diagnostic":
        target = rest[0] if rest else None
        await _diagnose(model_manager, target)
        return

    if sub == "set" and len(rest) >= 3:
        _mutate(model_manager, *rest[:3])
        return

    # otherwise treat first token as provider name (optional model)
    new_prov = sub
    maybe_model = rest[0] if rest else None
    await _switch_provider(model_manager, new_prov, maybe_model, context)


async def _diagnose(model_manager: ModelManager, target: str | None) -> None:
    """Probe providers using chuk-llm's validation system."""
    if target:
        providers_to_test = [target] if model_manager.validate_provider(target) else []
        if not providers_to_test:
            console.print(f"[red]Unknown provider:[/red] {target}")
            return
    else:
        providers_to_test = model_manager.list_providers()

    tbl = Table(title="Provider diagnostics")
    tbl.add_column("Provider", style="green")
    tbl.add_column("Status", style="cyan")
    tbl.add_column("Features", style="yellow")
    tbl.add_column("Default Model", style="blue")

    # Get provider data using same source as working functions
    try:
        all_providers_data = model_manager.list_available_providers()
    except Exception as e:
        console.print(f"[red]Error getting provider data:[/red] {e}")
        return

    for provider in providers_to_test:
        try:
            validation = model_manager.validate_provider_setup(provider)
            info = all_providers_data.get(provider, {})
            
            # Skip if provider has errors
            if "error" in info:
                tbl.add_row(provider, f"[red]✗ Error[/red]", "-", info["error"][:20] + "...")
                continue
            
            # FIXED: Use proper status logic
            status_icon, status_reason = _get_provider_status(provider, info)
            
            if status_icon == "✅":
                status = f"[green]{status_icon} Ready[/green]"
            else:
                status = f"[red]{status_icon} {status_reason}[/red]"
            
            # Features - check baseline_features for chuk-llm 0.7 compatibility
            baseline_features = info.get("baseline_features", [])
            features = []
            if "streaming" in baseline_features:
                features.append("📡")
            if "tools" in baseline_features:
                features.append("🔧")
            if "vision" in baseline_features:
                features.append("👁️")
            feature_str = "".join(features) or "📝"
            
            # Default model - use same data source as working functions
            default_model = info.get("default_model", "unknown")
            
            tbl.add_row(provider, status, feature_str, default_model)
            
        except Exception as exc:
            tbl.add_row(provider, f"[red]✗ Error[/red]", "-", str(exc)[:30] + "...")

    console.print(tbl)


def _render_list(model_manager: ModelManager) -> None:
    """List all available providers with comprehensive info."""
    tbl = Table(title="Available Providers")
    tbl.add_column("Provider", style="green")
    tbl.add_column("Status", style="cyan")
    tbl.add_column("Default Model", style="yellow")
    tbl.add_column("Models Available", style="blue")
    tbl.add_column("Features", style="magenta")

    current = model_manager.get_active_provider()
    
    try:
        all_providers_info = model_manager.list_available_providers()
    except Exception as e:
        console.print(f"[red]Error getting provider list:[/red] {e}")
        return

    for name, info in all_providers_info.items():
        if "error" in info:
            tbl.add_row(name, "[red]Error[/red]", "-", "-", info["error"][:30] + "...")
            continue
        
        # Mark current provider
        provider_name = f"[bold]{name}[/bold]" if name == current else name
        
        # FIXED: Use proper status logic instead of just checking has_api_key
        status_icon, status_reason = _get_provider_status(name, info)
        
        # FIXED: Models count - chuk-llm 0.7 uses "models" key
        models = info.get("models", [])
        if not models:
            models = info.get("available_models", [])  # fallback for older versions
        model_count = len(models)
        models_str = f"{model_count} models" if model_count > 0 else "No models found"
        
        # Features summary
        baseline_features = info.get("baseline_features", [])
        feature_icons = []
        if "streaming" in baseline_features:
            feature_icons.append("📡")
        if "tools" in baseline_features:
            feature_icons.append("🔧")
        if "vision" in baseline_features:
            feature_icons.append("👁️")
        features_str = "".join(feature_icons) or "📝"
        
        tbl.add_row(
            provider_name,
            status_icon,
            info.get("default_model", "-"),
            models_str,
            features_str
        )

    console.print(tbl)


def _render_config(model_manager: ModelManager) -> None:
    """Show detailed configuration for all providers."""
    tbl = Table(title="Provider Configurations")
    tbl.add_column("Provider", style="green")
    tbl.add_column("Setting", style="cyan")
    tbl.add_column("Value", style="yellow")

    try:
        all_providers_info = model_manager.list_available_providers()
    except Exception as e:
        console.print(f"[red]Error getting provider configuration:[/red] {e}")
        return
    
    for provider_name, info in all_providers_info.items():
        if "error" in info:
            tbl.add_row(provider_name, "error", info["error"][:50] + "...")
            continue
        
        # FIXED: Get model count using correct key
        models = info.get("models", info.get("available_models", []))
        model_count = len(models)
        
        # FIXED: Use proper status logic for configuration display
        status_icon, status_reason = _get_provider_status(provider_name, info)
        
        settings = [
            ("api_base", info.get("api_base") or "default"),
            ("status", f"{status_icon} ({status_reason})"),
            ("default_model", info.get("default_model", "-")),
            ("model_count", str(model_count)),
            ("discovery_enabled", "✅" if info.get("discovery_enabled") else "❌"),
        ]
        
        for i, (setting, value) in enumerate(settings):
            provider_display = provider_name if i == 0 else ""
            tbl.add_row(provider_display, setting, str(value))

    console.print(tbl)


def _mutate(model_manager: ModelManager, prov: str, key: str, val: str) -> None:
    """
    Update provider configuration.
    
    Examples:
        CLI Mode:
            mcp-cli provider set openai api_key sk-your-key-here
            mcp-cli provider set anthropic api_base https://api.anthropic.com  
            mcp-cli provider set groq default_model llama-3.3-70b-versatile
            mcp-cli provider set mistral api_key null  # Remove API key
        
        Chat Mode:
            /provider set openai api_key sk-your-key-here
            /provider set anthropic api_base https://api.anthropic.com
            /provider set groq default_model llama-3.3-70b-versatile
            /provider set mistral api_key null  # Remove API key
    
    Supported keys:
        - api_key: Provider API key
        - api_base: Custom API endpoint URL
        - default_model: Default model for the provider
        - Any other provider-specific configuration
    """
    val_processed = None if val.lower() in {"none", "null"} else val
    
    try:
        # Use configure_provider for known settings
        if key == "api_key":
            model_manager.configure_provider(prov, api_key=val_processed)
        elif key == "api_base":
            model_manager.configure_provider(prov, api_base=val_processed)
        elif key == "default_model":
            model_manager.configure_provider(prov, default_model=val_processed)
        else:
            # Generic setting update
            model_manager.configure_provider(prov, **{key: val_processed})
        
        console.print(f"[green]Updated {prov}.{key}[/green]")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")


async def _switch_provider(
    model_manager: ModelManager,
    prov: str,
    model: str | None,
    ctx: Dict,
) -> None:
    """
    Switch provider with validation.
    
    Examples:
        CLI Mode:
            mcp-cli provider anthropic                    # Switch to Anthropic (default model)
            mcp-cli provider openai gpt-4o               # Switch to OpenAI with specific model  
            mcp-cli provider groq llama-3.3-70b-versatile # Switch to Groq with specific model
        
        Chat Mode:
            /provider anthropic                       # Switch to Anthropic (default model)
            /provider openai gpt-4o                  # Switch to OpenAI with specific model
            /provider groq llama-3.3-70b-versatile   # Switch to Groq with specific model
        
    The function will:
    1. Validate the provider exists
    2. Check provider configuration (API keys, etc.)
    3. Switch to the provider and model
    4. Update the chat/CLI context
    """
    if not model_manager.validate_provider(prov):
        available = ", ".join(model_manager.list_providers())
        console.print(f"[red]Unknown provider:[/red] {prov}")
        console.print(f"[yellow]Available:[/yellow] {available}")
        return

    # Get target model
    if model:
        target_model = model
    else:
        # Get default model for provider
        try:
            target_model = model_manager.get_default_model(prov)
            if not target_model:
                # Fallback - get any available model
                available_models = model_manager.get_available_models(prov)
                target_model = available_models[0] if available_models else "unknown"
        except Exception:
            target_model = "unknown"
    
    console.print(f"[dim]Switching to provider '{prov}' (model '{target_model}')…[/dim]")

    # FIXED: Validate provider setup with proper status checking
    try:
        all_providers_info = model_manager.list_available_providers()
        provider_info = all_providers_info.get(prov, {})
        
        # Use our enhanced status logic
        status_icon, status_reason = _get_provider_status(prov, provider_info)
        
        if status_icon == "❌":
            console.print(f"[red]Provider setup issues:[/red] {status_reason}")
            
            if prov == "ollama":
                console.print(f"[yellow]Tip:[/yellow] Make sure Ollama is running: ollama serve")
            elif "No API key" in status_reason:
                console.print(f"[yellow]Tip:[/yellow] Set API key with: mcp-cli provider set {prov} api_key YOUR_KEY")
            return
            
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Could not validate provider setup: {e}")

    # Switch
    try:
        model_manager.switch_model(prov, target_model)
    except Exception as e:
        console.print(f"[red]Failed to switch provider:[/red] {e}")
        return

    # Update context
    try:
        ctx.update({
            "provider": prov,
            "model": target_model,
            "client": model_manager.get_client(),
            "model_manager": model_manager,
        })
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Could not update client: {e}")
    
    console.print(f"[green]Switched to {prov}[/green] (model: {target_model})")


# Sync wrapper for legacy CLI paths
def provider_action(args: List[str], *, context: Dict) -> None:
    """Blocking facade around provider_action_async."""
    from mcp_cli.utils.async_utils import run_blocking
    run_blocking(provider_action_async(args, context=context))