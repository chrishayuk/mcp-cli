#!/usr/bin/env python3
"""
Safe test script that handles server failures gracefully
"""

import asyncio
import json
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, 'src')

async def test_individual_server():
    """Test a single server safely."""
    # Try to load config and skip if not available
    try:
        from mcp_cli.cli_options import load_config
        config = load_config("server_config.json")
        
        if not config or "mcpServers" not in config or not config["mcpServers"]:
            pytest.skip("No servers configured in server_config.json")
        
        # Use the first available server
        server_name = list(config["mcpServers"].keys())[0]
        
    except Exception as e:
        pytest.skip(f"Could not load server configuration: {e}")

    # Original test logic with the server_name
    from mcp_cli.tools.manager import ToolManager
    import logging

    logger = logging.getLogger(__name__)

    print(f"\n🧪 Testing server: {server_name}")
    print("-" * 40)

    tm = None
    try:
        tm = ToolManager(
            config_file="server_config.json",
            servers=[server_name],
            tool_timeout=30.0
        )

        # Set a timeout for initialization
        success = await asyncio.wait_for(tm.initialize(), timeout=10.0)

        if not success:
            print(f"❌ {server_name}: Failed to initialize")
            return False

        print(f"✅ {server_name}: Initialized successfully")

        # Get tools
        tools = await tm.get_all_tools()
        print(f"🛠️ {server_name}: {len(tools)} tools available")

        # Show first few tools
        for i, tool in enumerate(tools[:3]):
            print(f"   {i+1}. {tool.namespace}.{tool.name}")

        if len(tools) > 3:
            print(f"   ... and {len(tools) - 3} more tools")

        return True

    except asyncio.TimeoutError:
        print(f"⏰ {server_name}: Initialization timeout")
        return False
    except Exception as e:
        print(f"❌ {server_name}: Error - {str(e)[:100]}...")
        return False
    finally:
        if tm:
            try:
                await asyncio.wait_for(tm.close(), timeout=5.0)
            except:
                pass  # Ignore cleanup errors
            
async def find_working_servers():
    """Find which servers work and which don't."""
    
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("🔍 Testing MCP Servers Individually")
    print("=" * 50)
    
    # Read config
    try:
        with open('server_config.json', 'r') as f:
            config = json.load(f)
        servers = list(config.get('mcpServers', {}).keys())
    except Exception as e:
        print(f"❌ Cannot read config: {e}")
        return
    
    print(f"📋 Found {len(servers)} servers in config:")
    for server in servers:
        print(f"   - {server}")
    
    working_servers = []
    broken_servers = []
    
    # Test each server
    for server_name in servers:
        try:
            works = await test_individual_server(server_name)
            if works:
                working_servers.append(server_name)
            else:
                broken_servers.append(server_name)
        except KeyboardInterrupt:
            print(f"\n⏹️ Testing interrupted at {server_name}")
            break
        except Exception as e:
            print(f"💥 Unexpected error testing {server_name}: {e}")
            broken_servers.append(server_name)
    
    # Summary
    print(f"\n📊 Summary")
    print("=" * 20)
    print(f"✅ Working servers ({len(working_servers)}):")
    for server in working_servers:
        print(f"   - {server}")
    
    print(f"\n❌ Broken servers ({len(broken_servers)}):")
    for server in broken_servers:
        print(f"   - {server}")
    
    # Recommendations
    print(f"\n🎯 Recommendations:")
    if working_servers:
        print(f"1. Use working servers for now: {', '.join(working_servers)}")
        print(f"   Example: mcp-cli chat --server {working_servers[0]}")
    
    if broken_servers:
        print(f"2. Fix broken servers or remove them from config")
        print(f"3. Check if these servers need API keys or special setup:")
        for server in broken_servers:
            server_config = config['mcpServers'].get(server, {})
            command = server_config.get('command', 'Unknown')
            print(f"   - {server}: {command}")
    
    # Test perplexity specifically
    if 'perplexity' in servers:
        print(f"\n🔍 Perplexity Server Analysis:")
        perplexity_config = config['mcpServers'].get('perplexity', {})
        print(f"   Command: {perplexity_config.get('command')}")
        print(f"   Args: {perplexity_config.get('args', [])}")
        
        env_vars = perplexity_config.get('env', {})
        print(f"   Environment variables: {list(env_vars.keys())}")
        
        # Check if API key is set
        api_key_var = next((k for k in env_vars.keys() if 'API_KEY' in k), None)
        if api_key_var:
            actual_value = os.getenv(api_key_var)
            if actual_value:
                print(f"   ✅ {api_key_var} is set")
            else:
                print(f"   ❌ {api_key_var} is not set in environment")
                print(f"   💡 Set it with: export {api_key_var}=your_api_key")

if __name__ == "__main__":
    asyncio.run(find_working_servers())