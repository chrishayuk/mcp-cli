# Debug script to diagnose model discovery issues
# Save as: debug_models.py

import os
import sys
from pathlib import Path

def check_chuk_llm_version():
    """Check which version of chuk-llm is installed."""
    try:
        import chuk_llm
        print(f"✅ chuk-llm version: {getattr(chuk_llm, '__version__', 'unknown')}")
        
        # Check what's available
        print("\n📦 Available chuk-llm modules:")
        
        # Check for 0.6 features
        try:
            from chuk_llm.configuration.unified_config import get_config
            print("  ✅ unified_config (v0.6 feature)")
            
            config = get_config()
            providers = config.get_all_providers()
            print(f"  📋 Providers found: {len(providers)} - {providers}")
            
            # Check a specific provider
            if "openai" in providers:
                provider_config = config.get_provider("openai")
                print(f"  🔧 OpenAI models: {len(provider_config.models)} - {provider_config.models[:3]}...")
            
        except ImportError as e:
            print(f"  ❌ unified_config not available: {e}")
            print("  🔄 This suggests chuk-llm < 0.6")
        
        # Check for legacy features
        try:
            from chuk_llm.llm.llm_client import get_llm_client
            print("  ✅ legacy llm_client available")
        except ImportError:
            print("  ❌ legacy llm_client not available")
            
    except ImportError as e:
        print(f"❌ chuk-llm not installed: {e}")
        return False
    
    return True

def check_api_keys():
    """Check what API keys are configured."""
    print("\n🔑 API Keys Status:")
    
    keys_to_check = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "MISTRAL_API_KEY",
        "DEEPSEEK_API_KEY",
        "PERPLEXITY_API_KEY",
        "WATSONX_API_KEY"
    ]
    
    for key in keys_to_check:
        value = os.getenv(key)
        if value:
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"  ✅ {key}: {masked}")
        else:
            print(f"  ❌ {key}: not set")

def check_configuration_files():
    """Check what configuration files exist."""
    print("\n📁 Configuration Files:")
    
    # chuk-llm config locations
    chuk_config_dir = Path.home() / ".chuk_llm"
    mcp_config_dir = Path.home() / ".mcp-cli"
    
    config_files = [
        (chuk_config_dir / "config.yaml", "chuk-llm main config"),
        (chuk_config_dir / "providers.yaml", "chuk-llm provider overrides"),
        (chuk_config_dir / ".env", "chuk-llm environment"),
        (mcp_config_dir / "preferences.yaml", "MCP CLI preferences"),
        (mcp_config_dir / "preferences.json", "MCP CLI preferences (legacy)"),
        (mcp_config_dir / "models.json", "MCP CLI models (legacy)"),
    ]
    
    for file_path, description in config_files:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✅ {description}: {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {description}: {file_path} (not found)")

def test_client_creation():
    """Test creating clients directly."""
    print("\n🔧 Client Creation Test:")
    
    try:
        # Try chuk-llm 0.6 approach
        from chuk_llm.llm.client import get_client, list_available_providers
        
        print("  📋 Testing list_available_providers()...")
        providers = list_available_providers()
        
        for name, info in providers.items():
            if "error" in info:
                print(f"    ❌ {name}: {info['error']}")
            else:
                model_count = len(info.get("available_models", []))
                has_key = info.get("has_api_key", False)
                print(f"    {'✅' if has_key else '❌'} {name}: {model_count} models, API key: {has_key}")
        
        # Try creating a client
        print("\n  🔧 Testing client creation...")
        if "openai" in providers and providers["openai"].get("has_api_key"):
            try:
                client = get_client(provider="openai", model="gpt-4o-mini")
                print("    ✅ OpenAI client created successfully")
            except Exception as e:
                print(f"    ❌ OpenAI client creation failed: {e}")
        else:
            print("    ⚠️  Skipping OpenAI client test (no API key)")
            
    except ImportError:
        print("  ❌ chuk-llm 0.6 client functions not available")
        
        # Try legacy approach
        try:
            from chuk_llm.llm.llm_client import get_llm_client
            print("  🔄 Trying legacy client creation...")
            
            # This would need your existing ModelManager for config
            print("    ⚠️  Legacy client needs existing ModelManager")
            
        except ImportError as e:
            print(f"    ❌ Legacy client also not available: {e}")

def suggest_fixes():
    """Suggest potential fixes based on what we found."""
    print("\n🔧 Suggested Fixes:")
    
    # Check chuk-llm version
    try:
        from chuk_llm.configuration.unified_config import get_config
        print("1. ✅ chuk-llm 0.6 features available")
        
        # Check if models are actually empty
        config = get_config()
        providers = config.get_all_providers()
        
        if providers:
            total_models = 0
            for provider_name in providers:
                try:
                    provider_config = config.get_provider(provider_name)
                    total_models += len(provider_config.models)
                except:
                    pass
            
            if total_models == 0:
                print("2. ⚠️  chuk-llm config found but no models loaded")
                print("   💡 Try: config.reload() or check YAML configuration")
            else:
                print(f"2. ✅ Found {total_models} total models in configuration")
                print("   🤔 Issue might be in MCP CLI's provider info retrieval")
        
    except ImportError:
        print("1. ❌ chuk-llm 0.6 not available")
        print("   💡 Try: pip install 'chuk-llm>=0.6.0'")
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("3. ❌ No OPENAI_API_KEY found")
        print("   💡 Set in environment or ~/.chuk_llm/.env")
    else:
        print("3. ✅ OPENAI_API_KEY configured")
    
    # Check config files
    chuk_config = Path.home() / ".chuk_llm"
    if not chuk_config.exists():
        print("4. ❌ No ~/.chuk_llm directory")
        print("   💡 chuk-llm might not be properly initialized")
        print("   💡 Try running a basic chuk-llm command first")
    else:
        print("4. ✅ ~/.chuk_llm directory exists")

def main():
    print("🔍 MCP CLI Model Discovery Diagnostic")
    print("=" * 50)
    
    check_chuk_llm_version()
    check_api_keys()
    check_configuration_files()
    test_client_creation()
    suggest_fixes()
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("If you see '0 models' but chuk-llm 0.6 is installed,")
    print("the issue is likely in how MCP CLI is calling list_available_providers()")
    print("\nNext steps:")
    print("1. Check if chuk-llm 0.6 is actually installed")
    print("2. Verify API keys are set")
    print("3. Test chuk-llm directly outside of MCP CLI")

if __name__ == "__main__":
    main()


# Quick fix for the immediate "0 models" issue
# Add this to your ModelManager or provider command:

def debug_model_count_issue():
    """Debug why model count shows as 0."""
    print("\n🔍 Debugging model count issue...")
    
    try:
        from chuk_llm.llm.client import list_available_providers
        
        providers_info = list_available_providers()
        
        for name, info in providers_info.items():
            print(f"\nProvider: {name}")
            print(f"  Raw info keys: {list(info.keys())}")
            
            if "available_models" in info:
                models = info["available_models"]
                print(f"  available_models: {type(models)} with {len(models)} items")
                if models:
                    print(f"  First few models: {models[:3]}")
            else:
                print("  ❌ No 'available_models' key found")
            
            if "models" in info:
                models = info["models"] 
                print(f"  models: {type(models)} with {len(models)} items")
                if models:
                    print(f"  First few models: {models[:3]}")
            else:
                print("  ❌ No 'models' key found")
                
    except Exception as e:
        print(f"❌ Error in debug: {e}")

# Potential fix for the provider list command:
def fixed_render_list(model_manager):
    """Fixed version that handles missing model data better."""
    from rich.table import Table
    from mcp_cli.utils.rich_helpers import get_console
    
    console = get_console()
    
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
        
        # Status
        configured = info.get("has_api_key", False)
        status = "[green]✅[/green]" if configured else "[red]❌[/red]"
        
        # Models count - try multiple possible keys
        model_count = 0
        for key in ["available_models", "models", "model_list"]:
            if key in info and isinstance(info[key], list):
                model_count = len(info[key])
                break
        
        # Debug: print what we actually got
        if model_count == 0:
            print(f"Debug: {name} info keys: {list(info.keys())}")
        
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
            status,
            info.get("default_model", "-"),
            models_str,
            features_str
        )

    console.print(tbl)