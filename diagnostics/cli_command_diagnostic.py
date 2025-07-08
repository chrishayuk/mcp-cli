#!/usr/bin/env python3
"""
Simple, robust test script for MCP servers that avoids asyncio complications
"""

import json
import subprocess
import sys
import os
import time

def test_mcp_cli_command(command_args, timeout=30):
    """Test an mcp-cli command and return success status and output."""
    try:
        # Set up environment
        env = os.environ.copy()
        env['MCP_TOOL_TIMEOUT'] = str(timeout)
        env['CHUK_LOG_LEVEL'] = 'WARNING'  # Reduce noise
        
        # Run the command
        cmd = [sys.executable, '-m', 'mcp_cli'] + command_args
        print(f"🧪 Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        success = result.returncode == 0
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return False, "", f"Command failed: {e}"

def check_config_file(config_file="server_config.json"):
    """Check if config file exists and is valid."""
    print(f"📋 Checking config file: {config_file}")
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False, []
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        servers = list(config.get('mcpServers', {}).keys())
        print(f"✅ Config valid, found servers: {servers}")
        return True, servers
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config: {e}")
        return False, []
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False, []

def test_basic_commands():
    """Test basic mcp-cli commands."""
    print("\n" + "="*50)
    print("🧪 Testing Basic MCP CLI Commands")
    print("="*50)
    
    tests = [
        (["--help"], "Help command"),
        (["tools", "--help"], "Tools help"),
        (["servers", "--help"], "Servers help"),
        (["provider", "--help"], "Provider help"),
        (["models", "--help"], "Models help"),
    ]
    
    results = []
    for args, description in tests:
        print(f"\n📝 {description}")
        success, stdout, stderr = test_mcp_cli_command(args, timeout=10)
        
        if success:
            print(f"✅ PASS: {description}")
        else:
            print(f"❌ FAIL: {description}")
            if stderr:
                print(f"   Error: {stderr[:200]}...")
        
        results.append((description, success))
    
    return results

def test_server_commands(config_file="server_config.json"):
    """Test server-specific commands."""
    print("\n" + "="*50)
    print("🧪 Testing Server Commands")
    print("="*50)
    
    config_valid, servers = check_config_file(config_file)
    if not config_valid:
        return [("Config check", False)]
    
    results = [("Config check", True)]
    
    # Test commands that work with servers
    server_tests = [
        (["servers", "--config-file", config_file], "List servers"),
        (["tools", "--config-file", config_file], "List all tools"),
        (["prompts", "--config-file", config_file], "List prompts"),
        (["resources", "--config-file", config_file], "List resources"),
    ]
    
    for args, description in server_tests:
        print(f"\n📝 {description}")
        success, stdout, stderr = test_mcp_cli_command(args, timeout=45)
        
        if success:
            print(f"✅ PASS: {description}")
            # Show a preview of output
            if stdout:
                lines = stdout.strip().split('\n')
                # Filter out startup messages
                content_lines = [line for line in lines if not (
                    'MCP CLI ready' in line or 
                    'Available commands:' in line or 
                    'Use --help' in line or
                    line.startswith('  ')
                )]
                preview = content_lines[:3] if len(content_lines) > 3 else content_lines
                for line in preview:
                    if line.strip():  # Only show non-empty lines
                        print(f"   📄 {line[:80]}...")
                if len(content_lines) > 3:
                    print(f"   ... and {len(content_lines) - 3} more lines")
        else:
            print(f"❌ FAIL: {description}")
            if stderr:
                print(f"   Error: {stderr[:300]}...")
        
        results.append((description, success))
    
    # Test individual servers if we have them
    for server in servers[:2]:  # Test max 2 servers to keep it manageable
        print(f"\n📝 Testing server: {server}")
        success, stdout, stderr = test_mcp_cli_command(
            ["tools", "--config-file", config_file, "--server", server], 
            timeout=30
        )
        
        if success:
            print(f"✅ PASS: Server {server}")
            if "tool" in stdout.lower() or "Tool" in stdout:
                print(f"   📄 Tools found for {server}")
        else:
            print(f"❌ FAIL: Server {server}")
            if stderr:
                print(f"   Error: {stderr[:200]}...")
        
        results.append((f"Server {server}", success))
    
    return results

def test_provider_commands():
    """Test provider-related commands."""
    print("\n" + "="*50)
    print("🧪 Testing Provider Commands")
    print("="*50)
    
    provider_tests = [
        (["providers"], "Providers list (default)"),
        (["providers", "list"], "Providers list explicit"),
        (["provider"], "Provider status"),
        (["provider", "list"], "Provider list"),
        (["models"], "Models for current provider"),
        (["models", "openai"], "Models for openai"),
    ]
    
    results = []
    for args, description in provider_tests:
        print(f"\n📝 {description}")
        success, stdout, stderr = test_mcp_cli_command(args, timeout=15)
        
        if success:
            print(f"✅ PASS: {description}")
            if stdout:
                # Show first few lines, filtering out startup messages
                lines = stdout.strip().split('\n')
                content_lines = [line for line in lines if not (
                    'MCP CLI ready' in line or 
                    'Available commands:' in line or 
                    'Use --help' in line or
                    (line.startswith('  ') and 'commands:' in lines[max(0, lines.index(line)-1)])
                )]
                preview_lines = content_lines[:3] if content_lines else lines[:3]
                for line in preview_lines:
                    if line.strip():
                        print(f"   📄 {line[:80]}...")
        else:
            print(f"❌ FAIL: {description}")
            if stderr:
                print(f"   Error: {stderr[:200]}...")
        
        results.append((description, success))
    
    return results

def test_advanced_commands(config_file="server_config.json"):
    """Test more advanced commands if basic ones work."""
    print("\n" + "="*50)
    print("🧪 Testing Advanced Commands")
    print("="*50)
    
    advanced_tests = [
        (["ping", "--config-file", config_file], "Ping servers"),
        (["cmd", "--help"], "Command mode help"),
    ]
    
    results = []
    for args, description in advanced_tests:
        print(f"\n📝 {description}")
        success, stdout, stderr = test_mcp_cli_command(args, timeout=20)
        
        if success:
            print(f"✅ PASS: {description}")
            if stdout and "help" not in args:
                lines = stdout.strip().split('\n')[:2]
                for line in lines:
                    if line.strip():
                        print(f"   📄 {line[:80]}...")
        else:
            print(f"❌ FAIL: {description}")
            if stderr:
                print(f"   Error: {stderr[:200]}...")
        
        results.append((description, success))
    
    return results

def run_manual_tests():
    """Show manual tests the user can run."""
    print("\n" + "="*50)
    print("🔧 Manual Tests to Try")
    print("="*50)
    
    manual_tests = [
        "uv run mcp-cli --help",
        "uv run mcp-cli tools --config-file server_config.json",
        "uv run mcp-cli servers --config-file server_config.json", 
        "uv run mcp-cli providers list",
        "uv run mcp-cli models",
        "uv run mcp-cli ping --config-file server_config.json",
        "uv run mcp-cli chat --config-file server_config.json",
    ]
    
    print("Try these commands manually:")
    for i, test in enumerate(manual_tests, 1):
        print(f"   {i}. {test}")

def main():
    """Run all tests and provide summary."""
    print("🔍 Complete MCP CLI Test Suite")
    print("="*60)
    
    all_results = []
    
    # Run all test suites
    try:
        basic_results = test_basic_commands()
        all_results.extend(basic_results)
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        all_results.append(("Basic tests", False))
    
    try:
        server_results = test_server_commands()
        all_results.extend(server_results)
    except Exception as e:
        print(f"❌ Server tests failed: {e}")
        all_results.append(("Server tests", False))
    
    try:
        provider_results = test_provider_commands()
        all_results.extend(provider_results)
    except Exception as e:
        print(f"❌ Provider tests failed: {e}")
        all_results.append(("Provider tests", False))
    
    # Only run advanced tests if basic ones mostly work
    basic_pass_rate = sum(1 for _, success in basic_results if success) / len(basic_results)
    if basic_pass_rate >= 0.5:  # If 50%+ of basic tests pass
        try:
            advanced_results = test_advanced_commands()
            all_results.extend(advanced_results)
        except Exception as e:
            print(f"❌ Advanced tests failed: {e}")
            all_results.append(("Advanced tests", False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Complete Test Results Summary")
    print("="*60)
    
    passed = 0
    total = len(all_results)
    
    # Group results by category
    categories = {
        "Basic": [r for r in all_results if any(x in r[0] for x in ["Help", "help"])],
        "Server": [r for r in all_results if any(x in r[0] for x in ["servers", "tools", "Server", "Config", "prompts", "resources"])],
        "Provider": [r for r in all_results if any(x in r[0] for x in ["Provider", "Models", "provider"])],
        "Advanced": [r for r in all_results if any(x in r[0] for x in ["Ping", "Command", "ping", "cmd"])],
    }
    
    for category, results in categories.items():
        if results:
            print(f"\n{category} Commands:")
            for test_name, success in results:
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"   {test_name:<30} {status}")
                if success:
                    passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Provide specific guidance based on results
    if passed >= total * 0.8:
        print("\n🎉 Excellent! Your MCP CLI is working great!")
        print("💡 Ready for production use:")
        print("   uv run mcp-cli chat --config-file server_config.json")
        
    elif passed >= total * 0.5:
        print("\n🎯 Good! Most functionality is working!")
        print("💡 Try these working commands:")
        working_commands = [
            "uv run mcp-cli --help",
            "uv run mcp-cli tools --config-file server_config.json",
            "uv run mcp-cli providers",
        ]
        for cmd in working_commands:
            print(f"   {cmd}")
            
    else:
        print("\n❌ Several issues found - needs attention")
        print("🔧 Try these fixes:")
        fixes = [
            "pip install -e . --force-reinstall",
            "uv sync",
            "Check that server_config.json exists and is valid",
            "Verify MCP server dependencies are installed",
        ]
        for fix in fixes:
            print(f"   - {fix}")
    
    # Always show manual tests
    run_manual_tests()
    
    print(f"\n🔍 For detailed debugging:")
    print(f"   - Check logs with: uv run mcp-cli tools --config-file server_config.json --verbose")
    print(f"   - Test specific server: uv run mcp-cli tools --config-file server_config.json --server sqlite")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)