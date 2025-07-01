#!/usr/bin/env python3
# diagnostics/status_diagnostic.py
"""
Comprehensive provider status diagnostic script.
Analyzes why providers show incorrect ✅/❌ status.
"""

import os
import sys
from pathlib import Path
import subprocess

def add_src_to_path():
    """Add src directory to Python path."""
    src_path = Path(__file__).parent.parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

def check_ollama_status():
    """Check if Ollama is running and has models."""
    print("\n🦙 Ollama Status Check:")
    
    try:
        # Check if ollama command exists
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            models = [line for line in result.stdout.split('\n') if line.strip() and not line.startswith('NAME')]
            model_count = len([m for m in models if m.strip()])
            print(f"  ✅ Ollama is running with {model_count} models")
            print(f"  📋 Models: {[m.split()[0] for m in models if m.strip()]}")
            return True, model_count
        else:
            print(f"  ❌ Ollama command failed: {result.stderr}")
            return False, 0
    except FileNotFoundError:
        print("  ❌ Ollama not installed or not in PATH")
        return False, 0
    except Exception as e:
        print(f"  ❌ Error checking Ollama: {e}")
        return False, 0

def check_api_keys():
    """Check all provider API keys."""
    print("\n🔑 API Key Status:")
    
    keys_to_check = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", 
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "watsonx": "WATSONX_API_KEY"
    }
    
    key_status = {}
    for provider, env_var in keys_to_check.items():
        value = os.getenv(env_var)
        has_key = bool(value)
        key_status[provider] = has_key
        
        if has_key:
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"  ✅ {provider}: {env_var} = {masked}")
        else:
            print(f"  ❌ {provider}: {env_var} not set")
    
    return key_status

def check_provider_info():
    """Check what ModelManager reports for each provider."""
    print("\n📊 ModelManager Provider Info:")
    
    try:
        from mcp_cli.model_manager import ModelManager
        manager = ModelManager()
        
        # Get all providers
        providers = manager.list_providers()
        print(f"  📋 Found {len(providers)} providers: {providers}")
        
        status_details = {}
        
        for provider in providers:
            print(f"\n  🔍 {provider}:")
            try:
                info = manager.get_provider_info(provider)
                
                # Key status indicators
                has_api_key = info.get("has_api_key", False)
                api_base = info.get("api_base")
                default_model = info.get("default_model", "None")
                model_count = len(info.get("models", []))
                
                print(f"    has_api_key: {has_api_key}")
                print(f"    api_base: {api_base}")
                print(f"    default_model: {default_model}")
                print(f"    models: {model_count}")
                
                # Check additional status indicators
                if "discovery_enabled" in info:
                    print(f"    discovery_enabled: {info['discovery_enabled']}")
                
                if "baseline_features" in info:
                    features = info["baseline_features"]
                    print(f"    features: {features}")
                
                status_details[provider] = {
                    "has_api_key": has_api_key,
                    "api_base": api_base,
                    "model_count": model_count,
                    "default_model": default_model
                }
                
            except Exception as e:
                print(f"    ❌ Error getting info: {e}")
                status_details[provider] = {"error": str(e)}
        
        return status_details
        
    except Exception as e:
        print(f"  ❌ Error with ModelManager: {e}")
        return {}

def check_provider_validation():
    """Check provider validation status."""
    print("\n🔬 Provider Validation:")
    
    try:
        from mcp_cli.model_manager import ModelManager
        manager = ModelManager()
        
        providers = manager.list_providers()
        validation_results = {}
        
        for provider in providers[:5]:  # Check first 5 to avoid spam
            print(f"\n  🧪 {provider}:")
            try:
                # Check validation
                validation = manager.validate_provider_setup(provider)
                is_configured = manager.is_provider_configured(provider)
                
                print(f"    validate_provider_setup: {validation}")
                print(f"    is_provider_configured: {is_configured}")
                
                validation_results[provider] = {
                    "validation": validation,
                    "is_configured": is_configured
                }
                
            except Exception as e:
                print(f"    ❌ Validation error: {e}")
                validation_results[provider] = {"error": str(e)}
        
        return validation_results
        
    except Exception as e:
        print(f"  ❌ Error with validation: {e}")
        return {}

def analyze_status_logic():
    """Analyze the current status logic."""
    print("\n🎯 Status Logic Analysis:")
    
    try:
        from mcp_cli.model_manager import ModelManager
        manager = ModelManager()
        
        # Get the raw provider data
        all_providers = manager.list_available_providers()
        
        print("  Current status logic seems to use 'has_api_key' field.")
        print("  Let's see what should ACTUALLY determine status:\n")
        
        recommendations = {}
        
        for provider, info in all_providers.items():
            has_api_key = info.get("has_api_key", False)
            api_base = info.get("api_base")
            model_count = len(info.get("models", []))
            
            # Determine what status SHOULD be
            if provider == "ollama":
                # Ollama should be ✅ if it's running, regardless of API key
                ollama_running, _ = check_ollama_status() if provider == "ollama" else (False, 0)
                should_be = "✅" if ollama_running else "❌"
                reason = "Ollama running" if ollama_running else "Ollama not running"
            else:
                # Other providers need API keys
                should_be = "✅" if has_api_key else "❌"
                reason = "Has API key" if has_api_key else "No API key"
            
            current_status = "✅" if has_api_key else "❌"
            
            print(f"  {provider}:")
            print(f"    Current: {current_status} (has_api_key: {has_api_key})")
            print(f"    Should be: {should_be} ({reason})")
            print(f"    Models: {model_count}")
            
            if api_base:
                print(f"    API Base: {api_base}")
            
            recommendations[provider] = {
                "current": current_status,
                "should_be": should_be,
                "reason": reason
            }
            print()
        
        return recommendations
        
    except Exception as e:
        print(f"  ❌ Error analyzing status logic: {e}")
        return {}

def suggest_fixes(recommendations):
    """Suggest specific fixes for status issues."""
    print("\n🔧 Suggested Fixes:")
    
    issues_found = False
    
    for provider, data in recommendations.items():
        if data["current"] != data["should_be"]:
            issues_found = True
            print(f"\n  ❌ {provider}: Status incorrect")
            print(f"    Current: {data['current']}")
            print(f"    Should be: {data['should_be']}")
            print(f"    Reason: {data['reason']}")
            
            if provider == "ollama":
                print(f"    💡 Fix: Update status logic to check if Ollama is running")
                print(f"       Add: subprocess.run(['ollama', 'list']) check")
            else:
                if "No API key" in data["reason"]:
                    env_var = f"{provider.upper()}_API_KEY"
                    print(f"    💡 Fix: Set {env_var} environment variable")
                else:
                    print(f"    💡 Fix: Check provider configuration")
    
    if not issues_found:
        print("  ✅ All provider statuses appear correct!")
    else:
        print(f"\n  🎯 MAIN FIX NEEDED: Update provider status logic in commands/provider.py")
        print(f"     The status should consider:")
        print(f"     1. Ollama: Check if running (subprocess)")
        print(f"     2. Others: Check API key + validation")

def main():
    """Run comprehensive provider status diagnostic."""
    print("🔍 Provider Status Diagnostic")
    print("=" * 60)
    print("Analyzing why some providers show incorrect ✅/❌ status")
    
    # Add src to path
    add_src_to_path()
    
    # Run all checks
    ollama_status = check_ollama_status()
    api_key_status = check_api_keys()
    provider_info = check_provider_info()
    validation_results = check_provider_validation()
    recommendations = analyze_status_logic()
    
    # Provide recommendations
    suggest_fixes(recommendations)
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print("The main issue is likely that the status logic only checks 'has_api_key'")
    print("but doesn't account for local providers like Ollama or proper validation.")
    print("\nTo fix: Update the provider list command to use better status logic.")

if __name__ == "__main__":
    main()