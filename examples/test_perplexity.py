#!/usr/bin/env python3
"""
Complete diagnostic script for Perplexity MCP server issues
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

def check_command_exists(command):
    """Check if a command exists in PATH."""
    return shutil.which(command) is not None

def run_command(command, timeout=5):
    """Run a command safely with timeout."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, 
            text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -2, "", str(e)

def diagnose_perplexity():
    """Complete diagnostic for Perplexity server."""
    
    print("🔍 Perplexity MCP Server Diagnostics")
    print("=" * 50)
    
    # 1. Check Python environment
    print("1. Python Environment")
    print("-" * 20)
    print(f"   Python version: {sys.version}")
    print(f"   Python executable: {sys.executable}")
    
    # 2. Check if perplexity server is installed
    print("\n2. Perplexity Server Installation")
    print("-" * 35)
    
    perplexity_commands = [
        "mcp-server-perplexity",
        "perplexity-mcp-server",
        "python -m mcp_server_perplexity"
    ]
    
    working_command = None
    for cmd in perplexity_commands:
        if check_command_exists(cmd.split()[0]):
            print(f"   ✅ Found: {cmd}")
            working_command = cmd
            
            # Try to get version
            code, stdout, stderr = run_command(f"{cmd} --version")
            if code == 0:
                print(f"      Version: {stdout.strip()}")
            else:
                print(f"      Version check failed: {stderr}")
            break
        else:
            print(f"   ❌ Not found: {cmd}")
    
    if not working_command:
        print("\n   💡 Install perplexity server:")
        print("      pip install mcp-server-perplexity")
        print("      OR")
        print("      uv add mcp-server-perplexity")
        return False
    
    # 3. Check API key
    print("\n3. API Key Configuration")
    print("-" * 25)
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if api_key:
        print(f"   ✅ PERPLEXITY_API_KEY set ({len(api_key)} characters)")
        # Don't print the actual key for security
        print(f"      Starts with: {api_key[:8]}...")
    else:
        print("   ❌ PERPLEXITY_API_KEY not set")
        print("   💡 Get API key from: https://www.perplexity.ai/settings/api")
        print("   💡 Set with: export PERPLEXITY_API_KEY=your_key_here")
        return False
    
    # 4. Test server startup
    print("\n4. Server Startup Test")
    print("-" * 20)
    print("   🧪 Testing server startup...")
    
    # Try to start server and see if it responds
    code, stdout, stderr = run_command(f"{working_command} --help", timeout=10)
    if code == 0:
        print("   ✅ Server responds to --help")
        if stdout:
            print(f"      Help output: {stdout[:100]}...")
    else:
        print(f"   ❌ Server startup failed")
        print(f"      Error: {stderr}")
        return False
    
    # 5. Check server config
    print("\n5. Server Configuration")
    print("-" * 22)
    config_file = "server_config.json"
    if Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            perplexity_config = config.get('mcpServers', {}).get('perplexity')
            if perplexity_config:
                print("   ✅ Perplexity config found:")
                print(f"      Command: {perplexity_config.get('command')}")
                print(f"      Args: {perplexity_config.get('args', [])}")
                print(f"      Env vars: {list(perplexity_config.get('env', {}).keys())}")
            else:
                print("   ❌ No perplexity config in server_config.json")
                return False
        except Exception as e:
            print(f"   ❌ Error reading config: {e}")
            return False
    else:
        print(f"   ❌ {config_file} not found")
        return False
    
    # 6. Test with MCP CLI
    print("\n6. MCP CLI Integration Test")
    print("-" * 27)
    print("   🧪 Testing MCP CLI with perplexity...")
    
    # Set environment for test
    env = os.environ.copy()
    env['MCP_TOOL_TIMEOUT'] = '30'  # Short timeout for test
    
    test_cmd = f"{sys.executable} -m mcp_cli tools list --server perplexity"
    try:
        result = subprocess.run(
            test_cmd, shell=True, capture_output=True, 
            text=True, timeout=15, env=env, cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("   ✅ MCP CLI can connect to perplexity server")
            # Count tools mentioned in output
            if "tool" in result.stdout.lower():
                print("   ✅ Tools are available")
        else:
            print("   ❌ MCP CLI connection failed")
            print(f"      Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⏰ MCP CLI test timed out")
        return False
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False
    
    print("\n✅ All diagnostics passed! Perplexity server should work.")
    return True

def create_working_perplexity_config():
    """Create a working perplexity configuration."""
    
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        print("❌ Cannot create config without PERPLEXITY_API_KEY")
        return
    
    config = {
        "mcpServers": {
            "perplexity": {
                "command": "mcp-server-perplexity",
                "args": [],
                "env": {
                    "PERPLEXITY_API_KEY": api_key
                }
            }
        },
        "toolSettings": {
            "defaultTimeout": 300.0,
            "maxConcurrency": 4
        }
    }
    
    with open('server_config_perplexity.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created server_config_perplexity.json")
    print("\n🚀 Test with:")
    print("   export MCP_TOOL_TIMEOUT=300")
    print("   mcp-cli chat --config server_config_perplexity.json")

if __name__ == "__main__":
    print("🔧 Starting Perplexity diagnostics...\n")
    
    success = diagnose_perplexity()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Try: mcp-cli chat --server perplexity")
        print("2. Ask: 'what are the latest tech trends?'")
        
        create_working_perplexity_config()
    else:
        print("\n🔧 Fix the issues above, then run diagnostics again")
        print("\n💡 Quick setup commands:")
        print("   pip install chuk-mcp-perplexity")
        print("   export PERPLEXITY_API_KEY=your_api_key_here")
        print("   python perplexity_diagnostics.py")