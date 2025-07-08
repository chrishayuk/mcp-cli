#!/usr/bin/env python3
"""
Complete MCP CLI system diagnostics
"""

import os
import sys
import json
import subprocess
import importlib
from pathlib import Path

def check_python_packages():
    """Check required Python packages."""
    print("1. Python Package Dependencies")
    print("-" * 32)
    
    required_packages = [
        'mcp_cli',
        'chuk_tool_processor', 
        'chuk_mcp',
        'typer',
        'rich',
        'anyio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            module = importlib.import_module(package.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            location = getattr(module, '__file__', 'unknown')
            print(f"   ✅ {package}: {version}")
            print(f"      Location: {location}")
        except ImportError:
            print(f"   ❌ {package}: Not installed")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_mcp_cli_installation():
    """Check MCP CLI installation."""
    print("\n2. MCP CLI Installation")
    print("-" * 24)
    
    # Check if mcp-cli command exists
    try:
        result = subprocess.run(['which', 'mcp-cli'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ mcp-cli command found: {result.stdout.strip()}")
        else:
            print("   ❌ mcp-cli command not found in PATH")
    except:
        print("   ❌ Cannot check mcp-cli command")
    
    # Check if we can import mcp_cli
    try:
        import mcp_cli
        print(f"   ✅ mcp_cli module: {mcp_cli.__file__}")
        
        # Check for main components
        from mcp_cli.tools.manager import ToolManager
        print("   ✅ ToolManager importable")
        
        from mcp_cli.run_command import run_command
        print("   ✅ run_command importable")
        
    except ImportError as e:
        print(f"   ❌ Cannot import mcp_cli: {e}")
        return False
    
    return True

def check_server_configs():
    """Check server configuration files."""
    print("\n3. Server Configuration Files")
    print("-" * 31)
    
    config_files = [
        'server_config.json',
        'server_config_minimal.json', 
        'server_config_working.json'
    ]
    
    valid_configs = []
    
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                servers = list(config.get('mcpServers', {}).keys())
                print(f"   ✅ {config_file}: {len(servers)} servers")
                print(f"      Servers: {', '.join(servers)}")
                
                # Check timeout settings
                tool_settings = config.get('toolSettings', {})
                if 'defaultTimeout' in tool_settings:
                    print(f"      Timeout: {tool_settings['defaultTimeout']}s")
                
                valid_configs.append(config_file)
                
            except json.JSONDecodeError as e:
                print(f"   ❌ {config_file}: Invalid JSON - {e}")
            except Exception as e:
                print(f"   ❌ {config_file}: Error - {e}")
        else:
            print(f"   ⚠️ {config_file}: Not found")
    
    return len(valid_configs) > 0

def test_basic_functionality():
    """Test basic MCP CLI functionality."""
    print("\n4. Basic Functionality Test")
    print("-" * 28)
    
    # Test help command
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'mcp_cli', '--help'], 
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("   ✅ mcp-cli --help works")
        else:
            print(f"   ❌ mcp-cli --help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot run mcp-cli help: {e}")
        return False
    
    # Test tools list with a simple server
    try:
        env = os.environ.copy()
        env['MCP_TOOL_TIMEOUT'] = '30'
        
        result = subprocess.run(
            [sys.executable, '-m', 'mcp_cli', 'tools', 'list', '--server', 'sqlite'], 
            capture_output=True, text=True, timeout=15, env=env
        )
        if result.returncode == 0:
            print("   ✅ tools list works with sqlite")
            if "tool" in result.stdout.lower():
                print("   ✅ Tools are listed correctly")
        else:
            print(f"   ⚠️ tools list failed: {result.stderr}")
    except Exception as e:
        print(f"   ⚠️ Cannot test tools list: {e}")
    
    return True

def check_environment_variables():
    """Check relevant environment variables."""
    print("\n5. Environment Variables")
    print("-" * 25)
    
    env_vars = [
        'MCP_TOOL_TIMEOUT',
        'CHUK_TOOL_TIMEOUT',
        'CHUK_LOG_LEVEL',
        'PERPLEXITY_API_KEY',
        'OPENAI_API_KEY'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Hide sensitive values
            if 'API_KEY' in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"   ✅ {var}: {display_value}")
        else:
            print(f"   ⚠️ {var}: Not set")

def create_test_environment():
    """Create a minimal test environment."""
    print("\n6. Creating Test Environment")
    print("-" * 29)
    
    # Create minimal working config
    test_config = {
        "mcpServers": {
            "sqlite": {
                "command": "mcp-server-sqlite",
                "args": ["--db-path", "./test.db"]
            }
        },
        "toolSettings": {
            "defaultTimeout": 300.0,
            "maxConcurrency": 4
        }
    }
    
    with open('test_config.json', 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print("   ✅ Created test_config.json")
    
    # Create test script
    test_script = f'''#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, 'src')

os.environ['MCP_TOOL_TIMEOUT'] = '300'

import subprocess
result = subprocess.run([
    '{sys.executable}', '-m', 'mcp_cli', 'chat', 
    '--config', 'test_config.json'
], env=os.environ)
'''
    
    with open('test_mcp_cli.py', 'w') as f:
        f.write(test_script)
    
    print("   ✅ Created test_mcp_cli.py")
    print("\n   🚀 Run test with: python test_mcp_cli.py")

def main():
    """Run complete diagnostics."""
    print("🔍 Complete MCP CLI System Diagnostics")
    print("=" * 40)
    
    all_good = True
    
    # Run all checks
    all_good &= check_python_packages()
    all_good &= check_mcp_cli_installation()
    all_good &= check_server_configs()
    all_good &= test_basic_functionality()
    
    check_environment_variables()
    create_test_environment()
    
    # Summary
    print("\n" + "=" * 40)
    if all_good:
        print("✅ All core systems working!")
        print("\n🎯 Ready to use:")
        print("   export MCP_TOOL_TIMEOUT=300")
        print("   python test_mcp_cli.py")
    else:
        print("❌ Some issues found - fix them and run diagnostics again")
        print("\n🔧 Common fixes:")
        print("   pip install mcp-server-sqlite")
        print("   export MCP_TOOL_TIMEOUT=300")

if __name__ == "__main__":
    main()