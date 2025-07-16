#!/usr/bin/env python3
"""
MCP CLI Diagnostic Script

Diagnoses ChukLLM discovery and integration issues.
Run this to debug why discovery might not be working.

Usage:
    python diagnostic.py
    python diagnostic.py --provider ollama
    python diagnostic.py --verbose
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Set up environment BEFORE any chuk_llm imports
def setup_test_environment():
    """Set up test environment exactly like MCP CLI does."""
    env_vars = {
        "CHUK_LLM_DISCOVERY_ENABLED": "true",
        "CHUK_LLM_AUTO_DISCOVER": "true",
        "CHUK_LLM_DISCOVERY_ON_STARTUP": "true",
        "CHUK_LLM_DISCOVERY_TIMEOUT": "10",
        "CHUK_LLM_DISCOVERY_DEBUG": "true",  # Force debug for diagnostics
        "CHUK_LLM_OLLAMA_DISCOVERY": "true",
        "CHUK_LLM_OPENAI_DISCOVERY": "true",
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value


def check_environment() -> Dict[str, Any]:
    """Check environment variable setup."""
    relevant_vars = [k for k in os.environ.keys() if "CHUK_LLM" in k]
    
    return {
        "chuk_llm_vars": {k: os.environ[k] for k in relevant_vars},
        "discovery_enabled": os.getenv("CHUK_LLM_DISCOVERY_ENABLED"),
        "ollama_discovery": os.getenv("CHUK_LLM_OLLAMA_DISCOVERY"),
        "debug_mode": os.getenv("CHUK_LLM_DISCOVERY_DEBUG"),
    }


def check_ollama_direct() -> Dict[str, Any]:
    """Check Ollama directly without ChukLLM."""
    try:
        import httpx
        
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return {
                    "available": True,
                    "models": models,
                    "model_count": len(models)
                }
            else:
                return {
                    "available": False,
                    "error": f"HTTP {response.status_code}"
                }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


def check_chuk_llm_import() -> Dict[str, Any]:
    """Check if ChukLLM imports correctly."""
    try:
        import chuk_llm
        return {
            "import_success": True,
            "version": getattr(chuk_llm, "__version__", "unknown"),
            "location": chuk_llm.__file__ if hasattr(chuk_llm, "__file__") else "unknown"
        }
    except Exception as e:
        return {
            "import_success": False,
            "error": str(e)
        }


def check_chuk_llm_providers() -> Dict[str, Any]:
    """Check ChukLLM provider configuration."""
    try:
        from chuk_llm.llm.client import list_available_providers
        
        providers = list_available_providers()
        return {
            "success": True,
            "providers": {
                name: {
                    "models": info.get("models", []),
                    "model_count": len(info.get("models", [])),
                    "has_api_key": info.get("has_api_key", False),
                    "discovery_enabled": info.get("discovery_enabled", False),
                    "discovery_stats": info.get("discovery_stats", {})
                }
                for name, info in providers.items()
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def check_chuk_llm_functions() -> Dict[str, Any]:
    """Check ChukLLM generated functions."""
    try:
        from chuk_llm.api.providers import list_provider_functions, has_function
        
        functions = list_provider_functions()
        
        # Check for specific functions
        test_functions = [
            "ask_ollama",
            "ask_ollama_llama3",
            "ask_ollama_llama3_3",
            "ask_openai",
            "ask_anthropic"
        ]
        
        function_status = {
            name: has_function(name) for name in test_functions
        }
        
        return {
            "success": True,
            "total_functions": len(functions),
            "sample_functions": functions[:10],
            "test_functions": function_status
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def test_basic_usage(provider: str = "ollama") -> Dict[str, Any]:
    """Test basic ChukLLM usage."""
    try:
        import chuk_llm
        
        # Try sync ask
        response = chuk_llm.ask_sync(
            "Say 'test successful'",
            provider=provider,
            max_tokens=50
        )
        
        return {
            "success": True,
            "provider": provider,
            "response_length": len(response),
            "response_preview": response[:100]
        }
    except Exception as e:
        return {
            "success": False,
            "provider": provider,
            "error": str(e)
        }


def run_discovery_test() -> Dict[str, Any]:
    """Test discovery refresh functionality."""
    try:
        from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
        
        print("🔄 Triggering Ollama discovery...")
        result = trigger_ollama_discovery_and_refresh()
        
        return {
            "success": True,
            "new_functions": len(result),
            "function_names": list(result.keys())[:10]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def print_section(title: str, data: Dict[str, Any], verbose: bool = False):
    """Print a diagnostic section."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and not verbose:
                # Summarize nested dicts unless verbose
                print(f"{key}: {len(value)} items")
            elif isinstance(value, list) and len(value) > 5 and not verbose:
                # Summarize long lists unless verbose
                print(f"{key}: {len(value)} items - {value[:3]}...")
            else:
                print(f"{key}: {value}")
    else:
        print(data)


def main():
    parser = argparse.ArgumentParser(description="MCP CLI ChukLLM Diagnostic")
    parser.add_argument("--provider", default="ollama", help="Provider to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test-discovery", action="store_true", help="Test discovery refresh")
    args = parser.parse_args()
    
    print("🚀 MCP CLI ChukLLM Diagnostic")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Setup environment (like MCP CLI does)
    setup_test_environment()
    print_section("Environment Setup", check_environment(), args.verbose)
    
    # Step 2: Check Ollama directly
    print_section("Ollama Direct Check", check_ollama_direct(), args.verbose)
    
    # Step 3: Check ChukLLM import
    print_section("ChukLLM Import", check_chuk_llm_import(), args.verbose)
    
    # Step 4: Check providers
    print_section("ChukLLM Providers", check_chuk_llm_providers(), args.verbose)
    
    # Step 5: Check generated functions
    print_section("ChukLLM Functions", check_chuk_llm_functions(), args.verbose)
    
    # Step 6: Test discovery if requested
    if args.test_discovery:
        print_section("Discovery Test", run_discovery_test(), args.verbose)
    
    # Step 7: Test basic usage
    print_section(f"Basic Usage Test ({args.provider})", test_basic_usage(args.provider), args.verbose)
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print('='*60)
    
    # Quick status checks
    ollama_status = check_ollama_direct()
    chuk_status = check_chuk_llm_import()
    provider_status = check_chuk_llm_providers()
    
    print(f"Ollama: {'✅' if ollama_status.get('available') else '❌'}")
    print(f"ChukLLM Import: {'✅' if chuk_status.get('import_success') else '❌'}")
    print(f"Provider Config: {'✅' if provider_status.get('success') else '❌'}")
    
    if ollama_status.get('available'):
        model_count = ollama_status.get('model_count', 0)
        print(f"Ollama Models: {model_count}")
    
    if provider_status.get('success'):
        providers = provider_status.get('providers', {})
        for name, info in providers.items():
            count = info.get('model_count', 0)
            discovery = info.get('discovery_enabled', False)
            print(f"{name}: {count} models, discovery: {'✅' if discovery else '❌'}")
    
    print(f"\n💡 Next steps:")
    if not ollama_status.get('available'):
        print("   - Start Ollama: ollama serve")
    if not chuk_status.get('import_success'):
        print("   - Check ChukLLM installation")
    if not provider_status.get('success'):
        print("   - Check ChukLLM configuration")
    if args.test_discovery:
        print("   - Check discovery test results above")
    else:
        print("   - Run with --test-discovery to test discovery")


if __name__ == "__main__":
    main()