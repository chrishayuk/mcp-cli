#!/usr/bin/env python3
"""
MCP Tool Registry Diagnostic Script - Brief Version

Focused diagnostic that checks tool registry and SSE transport health
without verbose tool listings.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the mcp_cli directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  dotenv not available, using system environment")

from mcp_cli.tools.manager import ToolManager

# Set up logging to see everything
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


async def diagnose_tool_registry(servers: List[str], config_file: str):
    """Complete diagnostic of tool registry and SSE transport health."""
    
    print("=" * 80)
    print("MCP TOOL REGISTRY DIAGNOSTIC (BRIEF)")
    print("=" * 80)
    print(f"Servers: {servers}")
    print(f"Config: {config_file}")
    
    # Check if config file exists
    config_path = Path(config_file)
    if config_path.exists():
        print(f"✅ Config file found: {config_file}")
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            print(f"Config contains {len(config_data.get('mcpServers', {}))} servers")
            
            # Show gateway server config (mask sensitive data)
            gateway_config = config_data.get('mcpServers', {}).get('gateway', {})
            if gateway_config:
                print("Gateway server configuration:")
                print(f"  Transport: {gateway_config.get('transport', 'not set')}")
                print(f"  URL: {gateway_config.get('url', 'not set')}")
                
                headers = gateway_config.get('headers', {})
                print(f"  Headers: {len(headers)} configured")
                for key in headers.keys():
                    if 'auth' in key.lower() or 'key' in key.lower() or 'token' in key.lower():
                        value = headers[key]
                        masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                        print(f"    {key}: {masked}")
                    else:
                        print(f"    {key}: {headers[key]}")
            else:
                print("❌ No gateway server configuration found")
                
        except Exception as e:
            print(f"❌ Error reading config: {e}")
    else:
        print(f"❌ Config file not found: {config_file}")
        print("Available config file locations to check:")
        possible_configs = [
            "~/.config/mcp-cli/config.json",
            "./config.json", 
            "../config.json",
            "config/config.json"
        ]
        for possible in possible_configs:
            expanded = Path(possible).expanduser()
            exists = "✅" if expanded.exists() else "❌"
            print(f"  {exists} {expanded}")
    
    # Check environment variables
    print("\nEnvironment Variables:")
    env_vars = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", 
        "MCP_TOOL_TIMEOUT", "CHUK_TOOL_TIMEOUT", "MCP_CLI_INIT_TIMEOUT"
    ]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask API keys for security
            if "API_KEY" in var:
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"  ✅ {var}={masked}")
            else:
                print(f"  ✅ {var}={value}")
        else:
            print(f"  ❌ {var}=<not set>")
    
    print()
    
    # Test gateway connectivity
    print("0. GATEWAY CONNECTIVITY TEST...")
    print("-" * 40)
    
    gateway_config = config_data.get('mcpServers', {}).get('gateway', {})
    gateway_url = gateway_config.get('url')
    
    if gateway_url:
        print(f"Testing connectivity to: {gateway_url}")
        try:
            import httpx
            
            headers = gateway_config.get('headers', {})
            print(f"  Using {len(headers)} headers")
            
            timeout = httpx.Timeout(10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    response = await client.get(gateway_url, headers=headers)
                    print(f"  Status: {response.status_code}")
                    print(f"  Headers: {dict(response.headers)}")
                    if response.status_code == 200:
                        text = response.text
                        print(f"  Response: {text[:200]}...")
                    elif response.status_code == 404:
                        print(f"  ⚠️  Gateway URL returns 404 - might be normal for MCP endpoint")
                    else:
                        print(f"  ⚠️  Unexpected status: {response.status_code}")
                        print(f"  Response: {response.text[:200]}...")
                    
                    print(f"  ✅ Gateway is reachable")
                    
                except httpx.TimeoutException:
                    print(f"  ❌ Gateway timeout after 10s")
                except httpx.ConnectError as e:
                    print(f"  ❌ Connection error: {e}")
                except Exception as e:
                    print(f"  ❌ Request error: {e}")
                
        except ImportError:
            print("  ⚠️  httpx not available for connectivity test")
        except Exception as e:
            print(f"  ❌ Connectivity test error: {e}")
    else:
        print("  ❌ No gateway URL configured")
    
    print()
    
    # Initialize tool manager
    print("1. INITIALIZING TOOL MANAGER...")
    print("-" * 40)
    
    tool_manager = ToolManager(
        config_file=config_file,
        servers=servers,
        tool_timeout=60.0  # Increase timeout to 60 seconds for search tools
    )
    
    success = await tool_manager.initialize()
    print(f"Initialization successful: {success}")
    
    if not success:
        print("❌ Tool manager initialization failed!")
        
        # Try to give more specific error information
        if not config_path.exists():
            print("\n🔧 SUGGESTED FIXES:")
            print("1. Create config file with your gateway server settings")
            print("2. Or specify correct config file path with --config")
            print("3. Check that your .env file contains required API keys")
        
        return
    
    print("✅ Tool manager initialized successfully")
    
    # Check SSE transport health
    print("\n1.5. SSE TRANSPORT HEALTH CHECK...")
    print("-" * 40)
    
    if hasattr(tool_manager, 'stream_manager') and tool_manager.stream_manager:
        stream_manager = tool_manager.stream_manager
        
        # Check if stream manager has SSE transport info
        if hasattr(stream_manager, '_transports') or hasattr(stream_manager, 'transport'):
            print("  Stream manager available - checking SSE health...")
            
            # Try to get SSE transport directly
            transport = None
            if hasattr(stream_manager, 'transport'):
                transport = stream_manager.transport
            elif hasattr(stream_manager, '_transports') and stream_manager._transports:
                # Get first transport (should be SSE)
                transport = list(stream_manager._transports.values())[0] if stream_manager._transports else None
            
            if transport:
                print(f"  Transport type: {type(transport).__name__}")
                print(f"  Transport connected: {getattr(transport, 'is_connected', lambda: 'unknown')()}")
                
                # Check SSE-specific attributes
                if hasattr(transport, 'session_id'):
                    print(f"  Session ID: {getattr(transport, 'session_id', 'None')}")
                if hasattr(transport, 'message_url'):
                    print(f"  Message URL: {getattr(transport, 'message_url', 'None')}")
                if hasattr(transport, '_initialized'):
                    print(f"  Initialized: {getattr(transport, '_initialized', 'unknown')}")
                if hasattr(transport, 'pending_requests'):
                    pending_count = len(getattr(transport, 'pending_requests', {}))
                    print(f"  Pending requests: {pending_count}")
                
                # Check metrics if available
                if hasattr(transport, 'get_metrics'):
                    try:
                        metrics = transport.get_metrics()
                        print("  Transport metrics:")
                        for key, value in metrics.items():
                            if value is not None:
                                print(f"    {key}: {value}")
                    except Exception as e:
                        print(f"  ❌ Error getting metrics: {e}")
                
                # Try ping test
                if hasattr(transport, 'send_ping'):
                    try:
                        print("  Testing transport ping...")
                        ping_result = await transport.send_ping()
                        print(f"  Ping result: {ping_result}")
                    except Exception as e:
                        print(f"  ❌ Ping failed: {e}")
            else:
                print("  ❌ No transport found in stream manager")
        else:
            print("  ❌ Stream manager has no transport information")
    else:
        print("  ❌ No stream manager available")
    
    print()

    # Get raw registry contents (brief)
    print("2. RAW REGISTRY CONTENTS...")
    print("-" * 40)
    
    if tool_manager._registry:
        try:
            registry_items = await asyncio.wait_for(
                tool_manager._registry.list_tools(),
                timeout=30.0
            )
            print(f"Total tools in registry: {len(registry_items)}")
            
            # Show just first few and last few tools
            if len(registry_items) > 0:
                print("Sample tools:")
                for i, (namespace, name) in enumerate(registry_items[:3], 1):
                    print(f"  {i}. namespace='{namespace}', name='{name}'")
                if len(registry_items) > 6:
                    print("  ...")
                if len(registry_items) > 3:
                    for i, (namespace, name) in enumerate(registry_items[-2:], len(registry_items)-1):
                        print(f"  {i}. namespace='{namespace}', name='{name}'")
                        
        except Exception as exc:
            print(f"❌ Error listing registry tools: {exc}")
    else:
        print("❌ No registry available")
    
    print()
    
    # Get all tools via ToolManager (brief)
    print("3. TOOL MANAGER SUMMARY...")
    print("-" * 40)
    
    all_tools = await tool_manager.get_all_tools()
    print(f"Total tools from get_all_tools(): {len(all_tools)}")
    
    unique_tools = await tool_manager.get_unique_tools()
    print(f"Total unique tools: {len(unique_tools)}")
    
    # Test LLM adaptation (brief)
    print(f"\n4. LLM ADAPTATION (OpenAI)...")
    print("-" * 40)
    
    try:
        llm_tools, name_mapping = await tool_manager.get_adapted_tools_for_llm("openai")
        print(f"Total LLM tools: {len(llm_tools)}")
        print(f"Name mapping entries: {len(name_mapping)}")
        print(f"Sample mapping: {dict(list(name_mapping.items())[:3])}")
        
    except Exception as e:
        print(f"❌ Error getting LLM tools: {e}")
    
    print()
    
    # Test tool resolution (brief)
    print("5. TOOL RESOLUTION TESTS...")
    print("-" * 40)
    
    # Test just 3 key tools
    test_names = ["mcp-grid-dev-echo", "mcp-grid-dev-duckduckgo-search", "mcp-grid-dev-google-search"]
    
    for test_name in test_names:
        try:
            namespace, resolved_name = await tool_manager._find_tool_in_registry(test_name)
            if namespace:
                print(f"✅ '{test_name}' -> {namespace}/{resolved_name}")
            else:
                print(f"❌ '{test_name}' -> Not found")
        except Exception as e:
            print(f"❌ '{test_name}' -> Error: {e}")
    
    print()
    
    # Test tool execution with fresh connections
    print("6. TOOL EXECUTION TEST...")
    print("-" * 40)
    
    # Test each tool individually with fresh connection
    test_tools = [
        {
            "name": "mcp-grid-dev-echo",
            "args": {"message": "diagnostic test"},
            "description": "Echo tool test"
        },
        {
            "name": "mcp-grid-dev-duckduckgo-search", 
            "args": {"query": "artificial intelligence", "max_results": 3, "snippet_words": 100},
            "description": "DuckDuckGo search test"
        },
        {
            "name": "mcp-grid-dev-google-search",
            "args": {"query": "machine learning", "max_results": 3, "snippet_words": 100}, 
            "description": "Google search test"
        }
    ]
    
    for i, test_tool in enumerate(test_tools):
        print(f"\nTest {i+1}: {test_tool['description']}: '{test_tool['name']}'")
        print(f"  Arguments: {test_tool['args']}")
        
        # Close existing tool manager to force fresh connection
        if i > 0:
            print("  Closing previous connection...")
            await tool_manager.close()
            
            # Create fresh tool manager
            print("  Creating fresh connection...")
            tool_manager = ToolManager(
                config_file=config_file,
                servers=servers,
                tool_timeout=60.0
            )
            
            success = await tool_manager.initialize()
            if not success:
                print("  ❌ Failed to reinitialize tool manager")
                continue
            print("  ✅ Fresh connection established")
        
        try:
            result = await tool_manager.execute_tool(test_tool['name'], test_tool['args'])
            print(f"  Success: {result.success}")
            
            # Check SSE transport health after each tool call
            if hasattr(tool_manager, 'stream_manager') and tool_manager.stream_manager:
                try:
                    stream_manager = tool_manager.stream_manager
                    transport = getattr(stream_manager, 'transport', None)
                    if transport and hasattr(transport, 'pending_requests'):
                        pending_count = len(getattr(transport, 'pending_requests', {}))
                        if pending_count > 0:
                            print(f"  ⚠️  {pending_count} pending requests after tool call")
                        
                        # Check if SSE task is still running
                        if hasattr(transport, 'sse_task'):
                            sse_task = getattr(transport, 'sse_task')
                            if sse_task and sse_task.done():
                                print(f"  ❌ SSE task died: {sse_task.exception()}")
                except Exception as e:
                    print(f"  Debug error: {e}")
            
            if result.error:
                print(f"  Error: {result.error}")
            
            if result.result:
                # Handle different result types
                if isinstance(result.result, dict):
                    if "error" in result.result:
                        print(f"  Server Error: {result.result['error']}")
                        if "available" in result.result:
                            print(f"  Tool Available: {result.result['available']}")
                        if "reason" in result.result:
                            print(f"  Failure Reason: {result.result['reason']}")
                    elif "results" in result.result:
                        # Search results
                        results = result.result["results"]
                        print(f"  Search Results: {len(results)} items")
                        for j, item in enumerate(results[:2], 1):  # Show first 2
                            title = item.get("title", "No title")[:50]
                            url = item.get("url", "No URL")[:50]
                            snippet = item.get("snippet", "No snippet")[:100]
                            print(f"    {j}. Title: {title}...")
                            print(f"       URL: {url}...")
                            print(f"       Snippet: {snippet}...")
                            print()
                    elif "message" in result.result:
                        # Echo result
                        print(f"  Echo Response: {result.result['message']}")
                    else:
                        # Other structured result - show more detail
                        if isinstance(result.result, dict):
                            # Pretty print dict structure
                            print(f"  Result Keys: {list(result.result.keys())}")
                            for key, value in list(result.result.items())[:3]:  # Show first 3 keys
                                value_str = str(value)[:100]
                                print(f"    {key}: {value_str}...")
                        else:
                            result_str = str(result.result)[:300]  # Show more characters
                            print(f"  Result: {result_str}...")
                else:
                    # Non-dict result
                    result_str = str(result.result)[:300]  # Show more characters
                    print(f"  Result: {result_str}...")
                    
            if result.execution_time:
                print(f"  Execution Time: {result.execution_time:.2f}s")
                
        except Exception as e:
            print(f"  ❌ Execution error: {e}")
            import traceback
            traceback.print_exc()
        
        print()  # Blank line between tests
    
    # Server info
    print("7. SERVER INFO...")
    print("-" * 40)
    
    try:
        server_info = await tool_manager.get_server_info()
        print(f"Connected servers: {len(server_info)}")
        for server in server_info:
            print(f"  {server.id}: {server.name} ({server.status}) - {server.tool_count} tools")
    except Exception as e:
        print(f"❌ Error getting server info: {e}")
    
    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    
    # Cleanup
    await tool_manager.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose MCP tool registry")
    parser.add_argument("--server", required=True, help="Server name to test")
    parser.add_argument("--config", help="Config file path (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Auto-detect config file if not specified
    if args.config:
        config_file = str(Path(args.config).expanduser())
    else:
        # Try common locations, starting with server_config.json in current directory
        possible_configs = [
            "./server_config.json",
            "~/.config/mcp-cli/config.json",
            "./config.json",
            "../config.json", 
            "config/config.json",
            "mcp_config.json"
        ]
        
        config_file = None
        for possible in possible_configs:
            expanded = Path(possible).expanduser()
            if expanded.exists():
                config_file = str(expanded)
                print(f"🔍 Auto-detected config: {config_file}")
                break
        
        if not config_file:
            print("❌ No config file found in common locations")
            print("Please specify config file with --config or create one of:")
            for possible in possible_configs:
                print(f"  {Path(possible).expanduser()}")
            sys.exit(1)
    
    servers = [args.server]
    
    asyncio.run(diagnose_tool_registry(servers, config_file))


if __name__ == "__main__":
    main()