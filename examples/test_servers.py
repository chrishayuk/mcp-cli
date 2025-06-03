#!/usr/bin/env python3
"""
Test script to check MCP server connectivity and list available tools
"""

import asyncio
import json
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, 'src')

async def test_servers():
    """Test MCP server connectivity."""
    
    from mcp_cli.tools.manager import ToolManager
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Read config
    try:
        with open('server_config.json', 'r') as f:
            config = json.load(f)
        servers = list(config.get('mcpServers', {}).keys())
        logger.info(f"📋 Found servers in config: {servers}")
    except FileNotFoundError:
        logger.error("❌ server_config.json not found")
        return
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in config: {e}")
        return
    
    # Test each server individually
    for server_name in servers:
        logger.info(f"\n🧪 Testing server: {server_name}")
        
        tm = ToolManager(
            config_file="server_config.json",
            servers=[server_name],  # Test one server at a time
            tool_timeout=30.0  # Shorter timeout for testing
        )
        
        try:
            # Initialize
            success = await tm.initialize()
            if not success:
                logger.error(f"❌ Failed to initialize {server_name}")
                continue
            
            logger.info(f"✅ {server_name} initialized successfully")
            
            # Get server info
            server_info = await tm.get_server_info()
            for info in server_info:
                logger.info(f"   📊 Server: {info.name}, Status: {info.status}, Tools: {info.tool_count}")
            
            # Get tools
            tools = await tm.get_all_tools()
            logger.info(f"   🛠️ Available tools: {len(tools)}")
            for tool in tools[:5]:  # Show first 5 tools
                logger.info(f"     - {tool.namespace}.{tool.name}: {tool.description[:50]}...")
            
            if len(tools) > 5:
                logger.info(f"     ... and {len(tools) - 5} more tools")
            
            # Test a simple tool call if available
            if tools:
                test_tool = tools[0]
                logger.info(f"   🧪 Testing tool: {test_tool.name}")
                
                # Create simple test arguments
                test_args = {}
                if test_tool.parameters and 'properties' in test_tool.parameters:
                    for prop_name, prop_info in test_tool.parameters['properties'].items():
                        if prop_info.get('type') == 'string':
                            test_args[prop_name] = "test query"
                            break
                
                if test_args:
                    try:
                        result = await tm.execute_tool(f"{test_tool.namespace}.{test_tool.name}", test_args)
                        if result.success:
                            logger.info(f"   ✅ Tool test successful")
                        else:
                            logger.warning(f"   ⚠️ Tool test failed: {result.error}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Tool test exception: {e}")
                else:
                    logger.info(f"   ⏭️ Skipping tool test (no suitable parameters)")
            
        except Exception as e:
            logger.error(f"❌ Error testing {server_name}: {e}")
        
        finally:
            await tm.close()
    
    # Test all servers together
    logger.info(f"\n🌐 Testing all servers together...")
    tm_all = ToolManager(
        config_file="server_config.json",
        servers=servers,
        tool_timeout=60.0
    )
    
    try:
        success = await tm_all.initialize()
        if success:
            logger.info(f"✅ All servers initialized successfully")
            
            tools = await tm_all.get_all_tools()
            logger.info(f"🛠️ Total tools available: {len(tools)}")
            
            # Group by server
            by_server = {}
            for tool in tools:
                if tool.namespace not in by_server:
                    by_server[tool.namespace] = []
                by_server[tool.namespace].append(tool)
            
            for server, server_tools in by_server.items():
                logger.info(f"   📊 {server}: {len(server_tools)} tools")
        else:
            logger.error(f"❌ Failed to initialize all servers")
    
    except Exception as e:
        logger.error(f"❌ Error testing all servers: {e}")
    
    finally:
        await tm_all.close()

if __name__ == "__main__":
    asyncio.run(test_servers())