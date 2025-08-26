# Debug script to diagnose model discovery issues
# diagnostics/debug_models.py

import os
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
                print(
                    f"  🔧 OpenAI models: {len(provider_config.models)} - {provider_config.models[:3]}..."
                )

        except ImportError as e:
            print(f"  ❌ unified_config not available: {e}")
            print("  🔄 This suggests chuk-llm < 0.6")

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
        "WATSONX_API_KEY",
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
                # FIXED: Use "models" key instead of "available_models" for chuk-llm 0.7
                model_count = len(info.get("models", info.get("available_models", [])))
                has_key = info.get("has_api_key", False)

                # Special handling for Ollama - check if it's running
                if name == "ollama":
                    import httpx

                    try:
                        response = httpx.get(
                            "http://localhost:11434/api/tags", timeout=2.0
                        )
                        if response.status_code == 200:
                            has_key = True  # Ollama running = no API key needed
                            ollama_models = response.json().get("models", [])
                            model_count = len(ollama_models)
                    except:
                        pass

                status = (
                    "✅" if has_key or (name == "ollama" and model_count > 0) else "❌"
                )
                print(f"    {status} {name}: {model_count} models, API key: {has_key}")

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

        print("1. ✅ chuk-llm 0.6+ features available")

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
                print("   🎯 ISSUE IDENTIFIED: MCP CLI looking for wrong key!")
                print("   💡 chuk-llm 0.7 uses 'models' key, not 'available_models'")

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


def debug_model_key_structure():
    """Debug exactly what keys chuk-llm 0.7 uses for models."""
    print("\n🔍 Key Structure Analysis:")

    try:
        from chuk_llm.llm.client import list_available_providers

        providers_info = list_available_providers()

        for name, info in list(providers_info.items())[:3]:  # Just first 3
            print(f"\n{name}:")
            print(f"  All keys: {list(info.keys())}")

            # Check each possible model key
            for key in ["models", "available_models", "model_list", "supported_models"]:
                if key in info:
                    models = info[key]
                    print(f"  ✅ {key}: {type(models)} with {len(models)} items")
                    if isinstance(models, list) and models:
                        print(f"    First few: {models[:3]}")
                else:
                    print(f"  ❌ {key}: not found")

    except Exception as e:
        print(f"❌ Error in key structure analysis: {e}")


def main():
    print("🔍 MCP CLI Model Discovery Diagnostic")
    print("=" * 50)

    check_chuk_llm_version()
    check_api_keys()
    check_configuration_files()
    test_client_creation()
    debug_model_key_structure()


if __name__ == "__main__":
    main()
