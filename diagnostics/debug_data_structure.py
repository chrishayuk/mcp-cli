# diagnostics/debug_data_structure.py
"""
Debug script to see exactly what chuk-llm 0.7 returns
"""

def debug_chuk_llm_0_7_structure():
    """Debug the actual data structure returned by chuk-llm 0.7"""
    from chuk_llm.llm.client import list_available_providers
    from chuk_llm.configuration.unified_config import get_config
    
    print("🔍 chuk-llm 0.7 Data Structure Analysis")
    print("=" * 50)
    
    # Test list_available_providers()
    print("📋 list_available_providers() structure:")
    providers = list_available_providers()
    
    for name, info in list(providers.items())[:3]:  # Just show first 3
        print(f"\n{name}:")
        print(f"  Keys: {list(info.keys())}")
        for key, value in info.items():
            if isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if value:
                    print(f"    Sample: {value[:2]}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys: {list(value.keys())}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
    
    print("\n" + "=" * 30)
    
    # Test direct config access
    print("🔧 Direct config access:")
    config = get_config()
    
    for provider_name in list(config.get_all_providers())[:3]:
        print(f"\n{provider_name} (direct config):")
        try:
            provider_config = config.get_provider(provider_name)
            print(f"  models: {len(provider_config.models)} items")
            print(f"  sample models: {provider_config.models[:3]}")
            print(f"  default_model: {provider_config.default_model}")
            print(f"  features: {[f.value for f in provider_config.features]}")
        except Exception as e:
            print(f"  Error: {e}")
    
    return providers, config

# Run this first to see the structure
if __name__ == "__main__":
    debug_chuk_llm_0_7_structure()


# ================================================================
# IMMEDIATE FIX: Add debug to your existing provider command
# ================================================================

def debug_and_fix_provider_list():
    """
    Add this function to your existing provider command to debug and fix the issue.
    """
    from chuk_llm.llm.client import list_available_providers
    from chuk_llm.configuration.unified_config import get_config
    
    print("\n🔍 DEBUG: Analyzing provider data...")
    
    # Get data both ways
    providers_via_client = list_available_providers()
    config = get_config()
    
    print("Results:")
    for provider_name in ["openai", "anthropic", "groq"]:
        print(f"\n{provider_name}:")
        
        # Via client function
        client_info = providers_via_client.get(provider_name, {})
        client_models = client_info.get("available_models", [])
        print(f"  Via client: {len(client_models)} models")
        
        # Via direct config
        try:
            provider_config = config.get_provider(provider_name)
            direct_models = provider_config.models
            print(f"  Via config: {len(direct_models)} models")
            print(f"  Sample models: {direct_models[:3]}")
        except Exception as e:
            print(f"  Config error: {e}")
    
    print("\n🔧 Conclusion: Use direct config access for accurate model counts!")

# ================================================================
# FIXED PROVIDER COMMAND FUNCTION
# ================================================================

def _render_list_fixed(model_manager):
    """Fixed version that gets models directly from chuk-llm config."""
    from rich.table import Table
    from mcp_cli.utils.rich_helpers import get_console
    from chuk_llm.configuration.unified_config import get_config
    
    console = get_console()
    
    tbl = Table(title="Available Providers")
    tbl.add_column("Provider", style="green")
    tbl.add_column("Status", style="cyan")
    tbl.add_column("Default Model", style="yellow")
    tbl.add_column("Models Available", style="blue")
    tbl.add_column("Features", style="magenta")

    current = model_manager.get_active_provider()
    
    try:
        # Get provider info via client (for status, etc.)
        all_providers_info = model_manager.list_available_providers()
        
        # Get config for direct model access
        config = get_config()
        
    except Exception as e:
        console.print(f"[red]Error getting provider list:[/red] {e}")
        return

    for name, info in all_providers_info.items():
        if "error" in info:
            tbl.add_row(name, "[red]Error[/red]", "-", "-", info["error"][:30] + "...")
            continue
        
        # Mark current provider
        provider_name = f"[bold]{name}[/bold]" if name == current else name
        
        # Status
        configured = info.get("has_api_key", False)
        status = "[green]✅[/green]" if configured else "[red]❌[/red]"
        
        # FIXED: Get models directly from config
        model_count = 0
        try:
            provider_config = config.get_provider(name)
            model_count = len(provider_config.models)
        except Exception as e:
            console.print(f"[yellow]Debug: {name} config error: {e}[/yellow]")
        
        models_str = f"{model_count} models" if model_count > 0 else "No models found"
        
        # Features summary - try to get from direct config too
        try:
            provider_config = config.get_provider(name)
            baseline_features = [f.value for f in provider_config.features]
        except:
            baseline_features = info.get("baseline_features", [])
        
        feature_icons = []
        if "streaming" in baseline_features:
            feature_icons.append("📡")
        if "tools" in baseline_features:
            feature_icons.append("🔧")
        if "vision" in baseline_features:
            feature_icons.append("👁️")
        features_str = "".join(feature_icons) or "📝"
        
        # FIXED: Get default model directly from config
        default_model = info.get("default_model", "-")
        try:
            provider_config = config.get_provider(name)
            if provider_config.default_model:
                default_model = provider_config.default_model
        except:
            pass
        
        tbl.add_row(
            provider_name,
            status,
            default_model,
            models_str,
            features_str
        )

    console.print(tbl)