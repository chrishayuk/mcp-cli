#!/usr/bin/env python3
# test_fixes.py - Test the provider command fixes

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

def test_ollama_detection():
    """Test the Ollama detection fix."""
    print("🦙 Testing Ollama Detection:")
    
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            model_lines = [line for line in lines[1:] if line.strip()]
            print(f"  ✅ Ollama running with {len(model_lines)} models")
            return True, len(model_lines)
        else:
            print(f"  ❌ Ollama command failed: {result.stderr}")
            return False, 0
    except Exception as e:
        print(f"  ❌ Ollama detection failed: {e}")
        return False, 0

def test_chuk_llm_keys():
    """Test which keys chuk-llm 0.7 actually uses."""
    print("\n🔑 Testing chuk-llm Key Structure:")
    
    try:
        from chuk_llm.llm.client import list_available_providers
        
        providers = list_available_providers()
        
        for name, info in list(providers.items())[:3]:
            print(f"\n  {name}:")
            
            # Check for different model keys
            models_key = None
            model_count = 0
            
            if "models" in info:
                models_key = "models"
                model_count = len(info["models"])
            elif "available_models" in info:
                models_key = "available_models"
                model_count = len(info["available_models"])
            
            print(f"    Model key: {models_key}")
            print(f"    Model count: {model_count}")
            print(f"    Has API key: {info.get('has_api_key', False)}")
            print(f"    Default model: {info.get('default_model', 'None')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ chuk-llm key test failed: {e}")
        return False

def test_provider_status_logic():
    """Test the improved provider status logic."""
    print("\n🎯 Testing Provider Status Logic:")
    
    try:
        from chuk_llm.llm.client import list_available_providers
        
        providers = list_available_providers()
        
        # Test the new status logic
        def get_provider_status(provider_name, provider_info):
            if provider_name.lower() == "ollama":
                try:
                    import subprocess
                    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        model_count = len([line for line in lines[1:] if line.strip()])
                        return "✅", f"Running ({model_count} models)"
                    return "❌", "Not running"
                except:
                    return "❌", "Not running"
            
            has_api_key = provider_info.get("has_api_key", False)
            if not has_api_key:
                return "❌", "No API key"
            
            model_count = len(provider_info.get("models", provider_info.get("available_models", [])))
            if model_count == 0:
                return "⚠️", "API key set but no models"
            
            return "✅", f"Configured ({model_count} models)"
        
        for name, info in providers.items():
            if "error" not in info:
                status_icon, status_reason = get_provider_status(name, info)
                print(f"  {name}: {status_icon} - {status_reason}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Status logic test failed: {e}")
        return False

def test_model_count_display():
    """Test the model count display logic."""
    print("\n📊 Testing Model Count Display:")
    
    try:
        from chuk_llm.llm.client import list_available_providers
        
        providers = list_available_providers()
        
        def get_model_count_display(provider_name, provider_info):
            if provider_name.lower() == "ollama":
                try:
                    import subprocess
                    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        count = len([line for line in lines[1:] if line.strip()])
                        return f"{count} models"
                    return "Ollama not running"
                except:
                    return "Ollama not running"
            
            models = provider_info.get("models", provider_info.get("available_models", []))
            count = len(models)
            
            if count == 0:
                return "No models found"
            elif count == 1:
                return "1 model"
            else:
                return f"{count} models"
        
        for name, info in list(providers.items())[:5]:  # Test first 5
            if "error" not in info:
                display = get_model_count_display(name, info)
                print(f"  {name}: {display}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model count display test failed: {e}")
        return False

def simulate_fixed_provider_list():
    """Simulate what the fixed provider list would look like."""
    print("\n📋 Simulated Fixed Provider List:")
    print("=" * 60)
    
    try:
        from chuk_llm.llm.client import list_available_providers
        
        providers = list_available_providers()
        
        # Headers
        print(f"{'Provider':<12} {'Status':<8} {'Default Model':<20} {'Models':<15} {'Features'}")
        print("-" * 70)
        
        for name, info in providers.items():
            if "error" in info:
                continue
                
            # Status
            if name.lower() == "ollama":
                try:
                    import subprocess
                    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        count = len([line for line in lines[1:] if line.strip()])
                        status = "✅"
                        models_display = f"{count} models"
                    else:
                        status = "❌"
                        models_display = "Not running"
                except:
                    status = "❌"
                    models_display = "Not running"
            else:
                has_key = info.get("has_api_key", False)
                status = "✅" if has_key else "❌"
                
                models = info.get("models", info.get("available_models", []))
                count = len(models)
                models_display = f"{count} models" if count > 0 else "No models"
            
            # Default model
            default = info.get("default_model", "-")
            if not default or default == "None":
                default = "-"
            
            # Features
            features = info.get("baseline_features", [])
            feature_icons = []
            if "streaming" in features:
                feature_icons.append("📡")
            if "tools" in features or "parallel_calls" in features:
                feature_icons.append("🔧")
            if "vision" in features:
                feature_icons.append("👁️")
            if "reasoning" in features:
                feature_icons.append("🧠")
            
            features_str = "".join(feature_icons) if feature_icons else "📝"
            
            print(f"{name:<12} {status:<8} {default:<20} {models_display:<15} {features_str}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Simulation failed: {e}")
        return False

def main():
    """Run all fix tests."""
    print("🧪 Testing Provider Command Fixes")
    print("=" * 50)
    
    tests = [
        ("Ollama Detection", test_ollama_detection),
        ("chuk-llm Keys", test_chuk_llm_keys),
        ("Provider Status Logic", test_provider_status_logic),
        ("Model Count Display", test_model_count_display),
        ("Fixed Provider List Simulation", simulate_fixed_provider_list),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary:")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All fixes are working correctly!")
        print("💡 Apply these changes to your provider.py and model_manager.py files")
    else:
        print(f"\n⚠️  {total - passed} tests failed - review the fixes")

if __name__ == "__main__":
    main()