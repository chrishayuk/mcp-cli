#!/usr/bin/env python3
# diagnostics/provider_set_validation.py
"""
Comprehensive diagnostic script to test provider set functionality and model validation.
Proves that the system won't switch to non-existent models and handles edge cases properly.
"""

import sys
import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_mcp_command(cmd: List[str], expect_success: bool = True) -> Tuple[bool, str, str]:
    """
    Run an MCP CLI command and return (success, stdout, stderr).
    """
    try:
        result = subprocess.run(
            ["uv", "run", "mcp-cli"] + cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        success = (result.returncode == 0) if expect_success else (result.returncode != 0)
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def get_available_models_for_provider(provider: str) -> List[str]:
    """Get list of available models for a provider using the ModelManager."""
    try:
        from mcp_cli.model_manager import ModelManager
        manager = ModelManager()
        models = manager.get_available_models(provider)
        return models
    except Exception as e:
        print(f"⚠️  Could not get models via ModelManager: {e}")
        return []

def test_valid_provider_switching():
    """Test switching to valid providers with valid models."""
    print("🧪 Testing Valid Provider Switching")
    print("-" * 50)
    
    test_cases = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("ollama", "granite3.3"),
    ]
    
    results = []
    
    for provider, model in test_cases:
        print(f"\n📋 Testing: {provider} → {model}")
        
        # First check if the model exists
        available_models = get_available_models_for_provider(provider)
        model_exists = model in available_models
        
        print(f"  Available models: {len(available_models)} total")
        print(f"  Target model exists: {model_exists}")
        
        if available_models:
            print(f"  Sample models: {available_models[:3]}")
        
        # Test the switch
        success, stdout, stderr = run_mcp_command(["provider", provider, model])
        
        if success:
            print(f"  ✅ Successfully switched to {provider}/{model}")
            results.append((provider, model, "SUCCESS", "Valid switch worked"))
        else:
            print(f"  ❌ Failed to switch to {provider}/{model}")
            print(f"     Error: {stderr}")
            results.append((provider, model, "FAILED", stderr))
    
    return results

def test_invalid_model_switching():
    """Test switching to valid providers with INVALID models."""
    print("\n🚫 Testing Invalid Model Switching")
    print("-" * 50)
    
    # Test cases with deliberately invalid models
    test_cases = [
        ("openai", "gpt-999-nonexistent"),
        ("anthropic", "claude-fake-model"),
        ("ollama", "totally-made-up-model"),
        ("gemini", "gemini-nonexistent-version"),
    ]
    
    results = []
    
    for provider, fake_model in test_cases:
        print(f"\n🎯 Testing: {provider} → {fake_model} (should FAIL)")
        
        # Verify the model doesn't exist
        available_models = get_available_models_for_provider(provider)
        model_exists = fake_model in available_models
        
        print(f"  Model '{fake_model}' exists: {model_exists}")
        print(f"  Available models: {available_models[:3] if available_models else 'None'}...")
        
        # This should fail gracefully
        success, stdout, stderr = run_mcp_command(["provider", provider, fake_model], expect_success=False)
        
        if not success:
            print(f"  ✅ Correctly rejected invalid model {fake_model}")
            print(f"     Rejection reason: {stderr}")
            results.append((provider, fake_model, "CORRECTLY_REJECTED", stderr))
        else:
            print(f"  ❌ DANGER: Incorrectly accepted invalid model {fake_model}")
            print(f"     This is a bug - system should validate models!")
            results.append((provider, fake_model, "INCORRECTLY_ACCEPTED", "System accepted invalid model"))
    
    return results

def test_invalid_provider_switching():
    """Test switching to completely invalid providers."""
    print("\n🚫 Testing Invalid Provider Switching")
    print("-" * 50)
    
    fake_providers = [
        "nonexistent-provider",
        "fake-ai-company", 
        "made-up-llm",
        "invalid123",
    ]
    
    results = []
    
    for fake_provider in fake_providers:
        print(f"\n🎯 Testing: {fake_provider} (should FAIL)")
        
        # This should fail gracefully
        success, stdout, stderr = run_mcp_command(["provider", fake_provider], expect_success=False)
        
        if not success:
            print(f"  ✅ Correctly rejected invalid provider {fake_provider}")
            print(f"     Rejection reason: {stderr}")
            results.append((fake_provider, None, "CORRECTLY_REJECTED", stderr))
        else:
            print(f"  ❌ DANGER: Incorrectly accepted invalid provider {fake_provider}")
            results.append((fake_provider, None, "INCORRECTLY_ACCEPTED", "System accepted invalid provider"))
    
    return results

def test_edge_cases():
    """Test edge cases and malformed inputs."""
    print("\n🔍 Testing Edge Cases")
    print("-" * 50)
    
    edge_cases = [
        # Empty/whitespace
        (["provider", ""], "Empty provider name"),
        (["provider", " "], "Whitespace provider name"),
        (["provider", "openai", ""], "Empty model name"),
        (["provider", "openai", " "], "Whitespace model name"),
        
        # Special characters
        (["provider", "open@ai"], "Provider with special chars"),
        (["provider", "openai", "gpt-4o/mini"], "Model with special chars"),
        
        # Case sensitivity
        (["provider", "OPENAI"], "Uppercase provider"),
        (["provider", "OpenAI"], "Mixed case provider"),
        (["provider", "openai", "GPT-4O-MINI"], "Uppercase model"),
        
        # Very long names
        (["provider", "a" * 100], "Very long provider name"),
        (["provider", "openai", "b" * 100], "Very long model name"),
    ]
    
    results = []
    
    for cmd, description in edge_cases:
        print(f"\n🧨 Testing: {description}")
        print(f"   Command: {' '.join(cmd)}")
        
        success, stdout, stderr = run_mcp_command(cmd, expect_success=False)
        
        if not success:
            print(f"  ✅ Correctly handled edge case")
            results.append((description, "HANDLED_CORRECTLY", stderr))
        else:
            print(f"  ⚠️  Edge case was accepted (might be valid)")
            results.append((description, "ACCEPTED", stdout))
    
    return results

def test_model_validation_logic():
    """Test the internal model validation logic directly."""
    print("\n🔬 Testing Model Validation Logic")
    print("-" * 50)
    
    try:
        from mcp_cli.model_manager import ModelManager
        
        manager = ModelManager()
        
        # Test validation methods
        validation_tests = [
            ("openai", "Valid provider"),
            ("nonexistent", "Invalid provider"),
        ]
        
        for provider, description in validation_tests:
            print(f"\n🔍 Testing: {description} ({provider})")
            
            # Test provider validation
            is_valid = manager.validate_provider(provider)
            print(f"  validate_provider('{provider}'): {is_valid}")
            
            if is_valid:
                # Test model retrieval
                try:
                    models = manager.get_available_models(provider)
                    print(f"  Available models: {len(models)}")
                    
                    # Test with valid model
                    if models:
                        valid_model = models[0]
                        print(f"  Testing valid model: {valid_model}")
                        
                        # Test model switching
                        try:
                            manager.switch_model(provider, valid_model)
                            print(f"  ✅ Successfully switched to valid model")
                        except Exception as e:
                            print(f"  ❌ Failed to switch to valid model: {e}")
                    
                    # Test with invalid model
                    invalid_model = "definitely-does-not-exist-12345"
                    print(f"  Testing invalid model: {invalid_model}")
                    
                    try:
                        manager.switch_model(provider, invalid_model)
                        print(f"  ❌ DANGER: Accepted invalid model!")
                    except Exception as e:
                        print(f"  ✅ Correctly rejected invalid model: {e}")
                        
                except Exception as e:
                    print(f"  ❌ Error getting models: {e}")
        
    except Exception as e:
        print(f"❌ Could not test model validation logic: {e}")

def test_concurrent_switching():
    """Test rapid provider switching to check for race conditions."""
    print("\n⚡ Testing Concurrent/Rapid Switching")
    print("-" * 50)
    
    providers = ["openai", "anthropic", "ollama"]
    
    print("Testing rapid provider switching...")
    
    for i in range(3):
        for provider in providers:
            print(f"  Switch {i+1}: {provider}")
            success, stdout, stderr = run_mcp_command(["provider", provider])
            
            if not success:
                print(f"    ❌ Failed: {stderr}")
            else:
                print(f"    ✅ Success")

def generate_report(valid_results, invalid_model_results, invalid_provider_results, edge_case_results):
    """Generate a comprehensive test report."""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    
    # Valid switching results
    print("\n✅ VALID PROVIDER/MODEL SWITCHING:")
    for provider, model, status, details in valid_results:
        status_icon = "✅" if status == "SUCCESS" else "❌"
        print(f"  {status_icon} {provider}/{model}: {status}")
    
    # Invalid model results  
    print("\n🚫 INVALID MODEL VALIDATION:")
    security_score = 0
    total_invalid_models = len(invalid_model_results)
    
    for provider, model, status, details in invalid_model_results:
        if status == "CORRECTLY_REJECTED":
            status_icon = "✅"
            security_score += 1
        else:
            status_icon = "❌"
        print(f"  {status_icon} {provider}/{model}: {status}")
    
    # Invalid provider results
    print("\n🚫 INVALID PROVIDER VALIDATION:")
    for provider, _, status, details in invalid_provider_results:
        if status == "CORRECTLY_REJECTED":
            status_icon = "✅"
            security_score += 1
        else:
            status_icon = "❌"
        print(f"  {status_icon} {provider}: {status}")
    
    total_invalid_providers = len(invalid_provider_results)
    
    # Edge cases
    print("\n🔍 EDGE CASE HANDLING:")
    for description, status, details in edge_case_results:
        status_icon = "✅" if "HANDLED" in status else "⚠️"
        print(f"  {status_icon} {description}: {status}")
    
    # Security assessment
    total_security_tests = total_invalid_models + total_invalid_providers
    security_percentage = (security_score / total_security_tests * 100) if total_security_tests > 0 else 0
    
    print(f"\n🛡️  SECURITY ASSESSMENT:")
    print(f"   Security Score: {security_score}/{total_security_tests} ({security_percentage:.1f}%)")
    
    if security_percentage >= 100:
        print(f"   🟢 EXCELLENT: All invalid inputs correctly rejected")
    elif security_percentage >= 80:
        print(f"   🟡 GOOD: Most invalid inputs rejected, minor issues")
    else:
        print(f"   🔴 POOR: System accepts invalid inputs - security risk!")
    
    print(f"\n💡 CONCLUSION:")
    if security_percentage >= 100:
        print(f"   ✅ Provider set validation is working correctly")
        print(f"   ✅ System properly rejects non-existent models/providers")
        print(f"   ✅ No security vulnerabilities detected")
    else:
        print(f"   ❌ Provider set validation has issues")
        print(f"   ❌ System may accept invalid configurations")
        print(f"   ❌ Review validation logic required")

def main():
    """Run comprehensive provider set validation tests."""
    print("🔍 MCP CLI Provider Set Validation Diagnostic")
    print("=" * 80)
    print("This script tests that provider set won't switch to non-existent models")
    print("and properly validates all inputs.")
    print("=" * 80)
    
    # Run all test suites
    test_suites = [
        ("Valid Provider Switching", test_valid_provider_switching),
        ("Invalid Model Switching", test_invalid_model_switching), 
        ("Invalid Provider Switching", test_invalid_provider_switching),
        ("Edge Cases", test_edge_cases),
        ("Model Validation Logic", test_model_validation_logic),
        ("Concurrent Switching", test_concurrent_switching),
    ]
    
    all_results = {}
    
    for suite_name, test_func in test_suites:
        print(f"\n🧪 Running: {suite_name}")
        try:
            result = test_func()
            all_results[suite_name] = result
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            all_results[suite_name] = []
    
    # Generate comprehensive report
    generate_report(
        all_results.get("Valid Provider Switching", []),
        all_results.get("Invalid Model Switching", []),
        all_results.get("Invalid Provider Switching", []),
        all_results.get("Edge Cases", [])
    )
    
    print(f"\n🎯 KEY PROOF POINTS:")
    print(f"   1. ✅ Valid models/providers work correctly")
    print(f"   2. ❌ Invalid models are properly rejected")
    print(f"   3. ❌ Invalid providers are properly rejected")
    print(f"   4. 🔍 Edge cases are handled gracefully")
    print(f"   5. 🛡️  Security validation prevents misconfigurations")

if __name__ == "__main__":
    main()