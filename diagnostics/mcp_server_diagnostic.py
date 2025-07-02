#!/usr/bin/env python3
# mcp_server_diagnostic.py - Comprehensive MCP server diagnostic and analysis tool

import sys
import os
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

def check_mcp_environment():
    """Check if MCP environment is properly set up."""
    print("🔍 MCP Environment Check:")
    print("=" * 50)
    
    # Check for chuk-mcp installation
    try:
        import chuk_mcp
        print(f"  ✅ chuk-mcp installed: {chuk_mcp.__file__}")
    except ImportError:
        print("  ❌ chuk-mcp not found - install with: pip install chuk-mcp")
        return False, None
    
    # Check for mcp-cli
    try:
        import mcp_cli
        print(f"  ✅ mcp-cli available: {mcp_cli.__file__}")
    except ImportError:
        print("  ❌ mcp-cli not found")
        return False, None
    
    # Check for config files in multiple locations and formats
    config_paths = [
        # Standard MCP locations
        Path.home() / ".config" / "mcp" / "config.json",
        Path.home() / ".mcp" / "config.json",
        Path.home() / "mcp_config.json",
        
        # mcp-cli default locations (from main.py)
        Path.cwd() / "server_config.json",  # Default from main.py
        Path.home() / "server_config.json",
        
        # Current directory variations
        Path.cwd() / "mcp_config.json",
        Path.cwd() / "config.json",
        Path.cwd() / ".mcp_config.json",
        
        # Common alternative locations
        Path.cwd() / "configs" / "mcp.json",
        Path.cwd() / "configs" / "server_config.json",
        Path.cwd() / "config" / "mcp.json",
        Path.cwd() / "config" / "server_config.json",
        
        # VS Code settings (often contains MCP config)
        Path.home() / ".vscode" / "settings.json",
        Path.cwd() / ".vscode" / "settings.json",
    ]
    
    config_file = None
    config_data = None
    
    print("  🔍 Searching for MCP configuration...")
    
    for config_path in config_paths:
        if config_path.exists():
            print(f"    📁 Checking: {config_path}")
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                
                # Check different config formats
                servers = None
                
                # Standard MCP format
                if "mcpServers" in data:
                    servers = data["mcpServers"]
                    print(f"    ✅ Found MCP config with {len(servers)} servers")
                    config_file = str(config_path)
                    config_data = data
                    break
                
                # VS Code settings format
                elif "mcp.serverConfigurations" in data:
                    servers = data["mcp.serverConfigurations"]
                    print(f"    ✅ Found VS Code MCP config with {len(servers)} servers")
                    # Convert VS Code format to standard format
                    config_data = {"mcpServers": servers}
                    config_file = str(config_path)
                    break
                
                # Alternative formats
                elif "servers" in data:
                    servers = data["servers"]
                    print(f"    ✅ Found alternative config format with {len(servers)} servers")
                    config_data = {"mcpServers": servers}
                    config_file = str(config_path)
                    break
                
                else:
                    print(f"    ⚠️  JSON file found but no MCP servers configured")
                    
            except json.JSONDecodeError as e:
                print(f"    ❌ Invalid JSON in {config_path}: {e}")
            except Exception as e:
                print(f"    ❌ Error reading {config_path}: {e}")
    
    if config_file:
        print(f"  ✅ Using config: {config_file}")
        return True, (config_file, config_data)
    else:
        print("  ❌ No valid MCP config file found")
        print("     Create a config file with MCP servers to analyze real connections")
        print()
        print("  📝 mcp-cli uses 'server_config.json' by default. Create one of:")
        print("     • server_config.json (in current directory)")
        print("     • ~/.config/mcp/config.json (standard MCP location)")
        print("     • mcp_config.json (alternative name)")
        print()
        print("  📋 Example server_config.json:")
        print('     {')
        print('       "mcpServers": {')
        print('         "sqlite": {')
        print('           "command": "mcp-server-sqlite",')
        print('           "args": ["--db-path", "example.db"]')
        print('         },')
        print('         "filesystem": {')
        print('           "command": "mcp-server-filesystem", ')
        print('           "args": ["--allowed-dir", "/Users/your-username/Documents"]')
        print('         }')
        print('       }')
        print('     }')
        print()
        print("  💡 Popular MCP servers to try:")
        print("     • mcp-server-sqlite: Database queries")
        print("     • mcp-server-filesystem: File operations")
        print("     • mcp-server-git: Git operations")
        print("     • mcp-server-brave-search: Web search")
        print("     • mcp-server-github: GitHub integration")
        print()
        print("  🔧 Install servers with:")
        print("     pip install mcp-server-sqlite")
        print("     pip install mcp-server-filesystem")
        return False, None

def check_available_mcp_servers():
    """Check which common MCP servers are available on the system."""
    print("\n🔍 Checking for installed MCP servers:")
    
    common_servers = {
        "mcp-server-sqlite": "SQLite database operations",
        "mcp-server-filesystem": "File system operations",
        "mcp-server-git": "Git repository operations", 
        "mcp-server-brave-search": "Web search capabilities",
        "mcp-server-github": "GitHub integration",
        "mcp-server-postgres": "PostgreSQL database operations",
        "mcp-server-memory": "Persistent memory/notes",
        "mcp-server-fetch": "HTTP requests and web scraping"
    }
    
    import shutil
    available_servers = []
    
    for server_cmd, description in common_servers.items():
        if shutil.which(server_cmd):
            print(f"  ✅ {server_cmd}: {description}")
            available_servers.append((server_cmd, description))
        else:
            print(f"  ❌ {server_cmd}: Not installed")
    
    if available_servers:
        print(f"\n  🎉 Found {len(available_servers)} installed MCP server(s)!")
        print("     You can create a config file using these servers.")
        return available_servers
    else:
        print("\n  ⚠️  No common MCP servers found in PATH")
        print("     Install some servers first:")
        print("     pip install mcp-server-sqlite mcp-server-filesystem")
        return []

def suggest_config_from_available_servers(available_servers):
    """Generate a suggested config based on available servers."""
    if not available_servers:
        return
        
    print("\n💡 Suggested server_config.json based on your installed servers:")
    print("=" * 60)
    print("{")
    print('  "mcpServers": {')
    
    suggestions = []
    for i, (server_cmd, description) in enumerate(available_servers):
        server_name = server_cmd.replace("mcp-server-", "")
        
        # Provide sensible default args for common servers
        if server_name == "sqlite":
            config = f'    "{server_name}": {{\n      "command": "{server_cmd}",\n      "args": ["--db-path", "./example.db"]\n    }}'
        elif server_name == "filesystem":
            import os
            home_docs = os.path.expanduser("~/Documents")
            config = f'    "{server_name}": {{\n      "command": "{server_cmd}",\n      "args": ["--allowed-dir", "{home_docs}"]\n    }}'
        elif server_name == "git":
            cwd = os.getcwd()
            config = f'    "{server_name}": {{\n      "command": "{server_cmd}",\n      "args": ["--repository", "{cwd}"]\n    }}'
        else:
            config = f'    "{server_name}": {{\n      "command": "{server_cmd}"\n    }}'
        
        suggestions.append(config)
    
    print(",\n".join(suggestions))
    print("\n  }")
    print("}")
    print("\n💾 Save this as 'server_config.json' in your current directory")
    print("   Then run the diagnostic again to test your servers!")

async def analyze_real_servers():
    """Connect to and analyze real MCP servers."""
    print("\n🌐 Real MCP Server Analysis:")
    print("=" * 50)
    
    # Check environment and get config
    env_ok, config_info = check_mcp_environment()
    if not env_ok or not config_info:
        print("  ⚠️  Cannot analyze real servers - no valid config found")
        
        # Check for available MCP servers and suggest config
        available_servers = check_available_mcp_servers()
        if available_servers:
            suggest_config_from_available_servers(available_servers)
        
        return create_mock_analysis()
    
    config_file, config_data = config_info
    
    try:
        from mcp_cli.tools.manager import ToolManager
        
        servers = config_data.get("mcpServers", {})
        print(f"  📋 Found {len(servers)} server(s) in config:")
        
        for name, server_config in servers.items():
            command = server_config.get("command", "unknown")
            args = server_config.get("args", [])
            env_vars = server_config.get("env", {})
            
            print(f"    • {name}:")
            print(f"      Command: {command}")
            if args:
                print(f"      Args: {args}")
            if env_vars:
                print(f"      Environment: {list(env_vars.keys())}")
        
        if not servers:
            print("  ⚠️  No servers configured in config file")
            return create_mock_analysis()
        
        # Create a temporary config file if needed (for ToolManager)
        temp_config_file = None
        if config_file.endswith("settings.json"):  # VS Code settings
            # Create temporary standard format config
            import tempfile
            temp_fd, temp_config_file = tempfile.mkstemp(suffix='.json')
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(config_data, f, indent=2)
            actual_config_file = temp_config_file
        else:
            actual_config_file = config_file
        
        # Verify server commands exist
        print(f"\n  🔍 Verifying server commands...")
        working_servers = []
        
        for name, server_config in servers.items():
            command = server_config.get("command", "")
            
            # Check if command exists
            import shutil
            if shutil.which(command):
                print(f"    ✅ {name}: Command '{command}' found")
                working_servers.append(name)
            else:
                print(f"    ❌ {name}: Command '{command}' not found in PATH")
                # Check if it's a file path
                if os.path.exists(command):
                    print(f"    ✅ {name}: Found as file path")
                    working_servers.append(name)
        
        if not working_servers:
            print(f"  ❌ No working servers found - all commands are missing")
            return create_mock_analysis()
        
        print(f"  📊 {len(working_servers)}/{len(servers)} servers have valid commands")
        
        # Try to initialize ToolManager
        print(f"\n  🔧 Initializing ToolManager...")
        
        tm = ToolManager(
            config_file=actual_config_file,
            servers=working_servers,  # Only use working servers
            tool_timeout=30.0
        )
        
        success = await tm.initialize()
        if not success:
            print("  ❌ Failed to initialize ToolManager")
            print("  💡 This might be due to:")
            print("     • Server scripts not executable")
            print("     • Missing dependencies for servers")
            print("     • Incorrect paths or arguments")
            return create_mock_analysis()
        
        print("  ✅ ToolManager initialized successfully")
        
        # Get server information
        try:
            server_info = await tm.get_server_info()
            print(f"  📊 Connected to {len(server_info)} server(s)")
        except Exception as e:
            print(f"  ❌ Failed to get server info: {e}")
            await tm.close()
            return create_mock_analysis()
        
        # Analyze each server
        analysis_results = []
        for i, srv in enumerate(server_info):
            print(f"\n  🔍 Analyzing server {i}: {srv.name}")
            
            analysis = await analyze_single_server(tm, i, srv, servers)
            analysis_results.append(analysis)
        
        await tm.close()
        
        # Clean up temp file if created
        if temp_config_file:
            try:
                os.unlink(temp_config_file)
            except:
                pass
        
        return analysis_results
        
    except Exception as e:
        print(f"  ❌ Analysis failed: {e}")
        import traceback
        print(f"  📝 Detailed error:")
        traceback.print_exc()
        return create_mock_analysis()

async def analyze_single_server(tm, server_index: int, server_info, server_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single MCP server in detail."""
    analysis = {
        "index": server_index,
        "name": server_info.name,
        "status": server_info.status,
        "tool_count": server_info.tool_count,
        "namespace": getattr(server_info, 'namespace', 'unknown'),
        "connection_test": "unknown",
        "protocol_version": "unknown",
        "capabilities": {},
        "tools": [],
        "features": {
            "tools": False,
            "resources": False,
            "prompts": False,
            "streaming": False,
            "notifications": False
        },
        "performance": {
            "ping_time": None,
            "tool_list_time": None
        },
        "config": server_configs.get(server_info.name, {})
    }
    
    try:
        # Test connection with ping
        start_time = time.perf_counter()
        streams = tm.get_streams()
        if server_index < len(streams):
            from chuk_mcp.protocol.messages import send_ping
            read_stream, write_stream = streams[server_index]
            
            try:
                ping_success = await asyncio.wait_for(
                    send_ping(read_stream, write_stream), 
                    timeout=5.0
                )
                ping_time = (time.perf_counter() - start_time) * 1000
                
                if ping_success:
                    analysis["connection_test"] = "✅ Success"
                    analysis["performance"]["ping_time"] = f"{ping_time:.1f}ms"
                    print(f"    ✅ Ping successful ({ping_time:.1f}ms)")
                else:
                    analysis["connection_test"] = "❌ Failed"
                    print(f"    ❌ Ping failed")
            except asyncio.TimeoutError:
                analysis["connection_test"] = "⏱️ Timeout"
                print(f"    ⏱️ Ping timeout (>5s)")
            except Exception as e:
                analysis["connection_test"] = f"❌ Error: {str(e)[:50]}"
                print(f"    ❌ Ping error: {e}")
        
        # Get tool list and measure performance
        start_time = time.perf_counter()
        try:
            all_tools = await tm.get_all_tools()
            tool_list_time = (time.perf_counter() - start_time) * 1000
            analysis["performance"]["tool_list_time"] = f"{tool_list_time:.1f}ms"
            
            # Debug: Show what tools we actually found
            print(f"    🔍 Debug: Found {len(all_tools)} total tools across all servers")
            
            # Try different ways to filter tools for this server
            server_tools = []
            
            # Method 1: Filter by namespace
            namespace_tools = [t for t in all_tools if getattr(t, 'namespace', '') == analysis['namespace']]
            print(f"    🔍 Debug: {len(namespace_tools)} tools match namespace '{analysis['namespace']}'")
            
            # Method 2: Filter by server name
            name_tools = [t for t in all_tools if analysis['name'] in getattr(t, 'namespace', '')]
            print(f"    🔍 Debug: {len(name_tools)} tools contain server name '{analysis['name']}'")
            
            # Method 3: If we're analyzing server N, take tools N*tools_per_server to (N+1)*tools_per_server
            estimated_tools_per_server = len(all_tools) // len(await tm.get_server_info()) if len(await tm.get_server_info()) > 0 else 0
            if estimated_tools_per_server > 0:
                start_idx = server_index * estimated_tools_per_server
                end_idx = min((server_index + 1) * estimated_tools_per_server, len(all_tools))
                indexed_tools = all_tools[start_idx:end_idx]
                print(f"    🔍 Debug: {len(indexed_tools)} tools by index estimation ({start_idx}-{end_idx})")
            else:
                indexed_tools = []
            
            # Use the method that gives us the most tools
            if len(namespace_tools) > 0:
                server_tools = namespace_tools
                print(f"    ✅ Using namespace-based tool filtering")
            elif len(name_tools) > 0:
                server_tools = name_tools
                print(f"    ✅ Using name-based tool filtering")
            elif len(indexed_tools) > 0:
                server_tools = indexed_tools
                print(f"    ✅ Using index-based tool filtering")
            else:
                # If nothing works, show some debug info
                print(f"    ⚠️  Could not match tools to server, showing debug info:")
                for i, tool in enumerate(all_tools[:3]):  # Show first 3 tools
                    print(f"      Tool {i}: name='{tool.name}', namespace='{getattr(tool, 'namespace', 'N/A')}'")
                if len(all_tools) > 3:
                    print(f"      ... and {len(all_tools) - 3} more tools")
            
            analysis["tools"] = [
                {
                    "name": t.name, 
                    "description": t.description,
                    "supports_streaming": getattr(t, 'supports_streaming', False)
                } 
                for t in server_tools
            ]
            analysis["features"]["tools"] = len(server_tools) > 0
            
            print(f"    🔧 Found {len(server_tools)} tools for this server ({tool_list_time:.1f}ms)")
            
            # Check for streaming support in tools
            streaming_tools = [t for t in server_tools if getattr(t, 'supports_streaming', False)]
            if streaming_tools:
                analysis["features"]["streaming"] = True
                print(f"    ⚡ {len(streaming_tools)} tools support streaming")
            
        except Exception as e:
            print(f"    ⚠️ Tool listing failed: {e}")
            import traceback
            print(f"    📝 Debug traceback:")
            traceback.print_exc()
        
        # Try to get resources
        try:
            if hasattr(tm, 'list_resources'):
                resources = await tm.list_resources()
                server_resources = [r for r in resources if r.get('server') == server_info.name]
                if server_resources:
                    analysis["features"]["resources"] = True
                    print(f"    📁 Found {len(server_resources)} resources")
        except Exception as e:
            print(f"    ⚠️ Resource listing failed: {e}")
        
        # Try to get prompts
        try:
            if hasattr(tm, 'list_prompts'):
                prompts = await tm.list_prompts()
                server_prompts = [p for p in prompts if p.get('server') == server_info.name]
                if server_prompts:
                    analysis["features"]["prompts"] = True
                    print(f"    💬 Found {len(server_prompts)} prompts")
        except Exception as e:
            print(f"    ⚠️ Prompt listing failed: {e}")
        
        # Try to get server details from stream manager and initialization data
        if hasattr(tm, 'stream_manager') and tm.stream_manager:
            try:
                print(f"    🔍 Attempting to get server initialization data...")
                
                # Try to get server data from stream manager
                server_data = None
                if hasattr(tm.stream_manager, 'get_server_data'):
                    server_data = tm.stream_manager.get_server_data(server_index)
                elif hasattr(tm.stream_manager, 'servers'):
                    servers_list = getattr(tm.stream_manager, 'servers', [])
                    if server_index < len(servers_list):
                        server_data = servers_list[server_index]
                
                if server_data:
                    print(f"    📋 Got server data: {list(server_data.keys())}")
                    analysis["protocol_version"] = server_data.get('protocol_version', 'unknown')
                    analysis["capabilities"] = server_data.get('capabilities', {})
                    
                    # Analyze capabilities
                    caps = analysis["capabilities"]
                    if caps:
                        analysis["features"].update({
                            "resources": analysis["features"]["resources"] or bool(caps.get("resources")),
                            "prompts": analysis["features"]["prompts"] or bool(caps.get("prompts")),
                            "streaming": analysis["features"]["streaming"] or bool(caps.get("tools", {}).get("streaming")),
                            "notifications": any([
                                caps.get("tools", {}).get("listChanged"),
                                caps.get("resources", {}).get("listChanged"),
                                caps.get("prompts", {}).get("listChanged")
                            ])
                        })
                        
                        print(f"    📋 Protocol: {analysis['protocol_version']}")
                        enabled_caps = [k for k, v in caps.items() if v]
                        if enabled_caps:
                            print(f"    🎯 Capabilities: {', '.join(enabled_caps)}")
                    else:
                        print(f"    ⚠️  No capabilities data in server info")
                else:
                    print(f"    ⚠️  Could not get server data from stream manager")
                    
                    # Try alternative method - direct stream inspection
                    streams = tm.get_streams()
                    if server_index < len(streams):
                        print(f"    🔍 Trying to inspect stream {server_index} directly...")
                        read_stream, write_stream = streams[server_index]
                        
                        # Try to get some basic MCP info
                        try:
                            from chuk_mcp.protocol.messages import send_tools_list
                            tools_response = await asyncio.wait_for(
                                send_tools_list(read_stream, write_stream),
                                timeout=5.0
                            )
                            if tools_response and "tools" in tools_response:
                                server_specific_tools = tools_response["tools"]
                                print(f"    🔧 Direct tools query: {len(server_specific_tools)} tools")
                                
                                # Update our analysis with the direct tools data
                                analysis["tools"] = [
                                    {
                                        "name": tool.get("name", "unknown"),
                                        "description": tool.get("description", ""),
                                        "supports_streaming": False  # Default for now
                                    }
                                    for tool in server_specific_tools
                                ]
                                analysis["features"]["tools"] = len(server_specific_tools) > 0
                                
                        except Exception as e:
                            print(f"    ⚠️  Direct tools query failed: {e}")
                    
            except Exception as e:
                print(f"    ⚠️ Could not get server details: {e}")
                import traceback
                print(f"    📝 Debug traceback:")
                traceback.print_exc()
        
        # Show config info
        config = analysis["config"]
        if config:
            print(f"    ⚙️  Config: {config.get('command', 'unknown')} {' '.join(config.get('args', []))}")
        
    except Exception as e:
        print(f"    ❌ Analysis failed: {e}")
        analysis["connection_test"] = f"❌ Error: {e}"
    
    return analysis

def create_mock_analysis() -> List[Dict[str, Any]]:
    """Create mock analysis when real servers aren't available."""
    print("\n  🎭 Creating mock analysis (no real servers available)")
    
    return [
        {
            "index": 0,
            "name": "mock-sqlite",
            "status": "Mock",
            "tool_count": 4,
            "namespace": "sqlite",
            "connection_test": "🎭 Mock",
            "protocol_version": "2025-06-18",
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"listChanged": True, "subscribe": True}
            },
            "tools": [
                {"name": "query", "description": "Execute SQL queries"},
                {"name": "schema", "description": "Get database schema"},
                {"name": "tables", "description": "List tables"},
                {"name": "insert", "description": "Insert data"}
            ],
            "features": {
                "tools": True,
                "resources": True,
                "prompts": False,
                "streaming": False,
                "notifications": True
            },
            "performance": {
                "ping_time": "12.3ms",
                "tool_list_time": "8.7ms"
            }
        },
        {
            "index": 1,
            "name": "mock-filesystem",
            "status": "Mock",
            "tool_count": 6,
            "namespace": "fs",
            "connection_test": "🎭 Mock",
            "protocol_version": "2025-06-18",
            "capabilities": {
                "tools": {"listChanged": True, "streaming": True},
                "resources": {"listChanged": True, "subscribe": True},
                "logging": {}
            },
            "tools": [
                {"name": "read_file", "description": "Read file contents"},
                {"name": "write_file", "description": "Write file contents"},
                {"name": "list_files", "description": "List directory contents"},
                {"name": "create_dir", "description": "Create directory"},
                {"name": "delete", "description": "Delete file/directory"},
                {"name": "copy", "description": "Copy file/directory"}
            ],
            "features": {
                "tools": True,
                "resources": True,
                "prompts": False,
                "streaming": True,
                "notifications": True
            },
            "performance": {
                "ping_time": "8.1ms",
                "tool_list_time": "15.2ms"
            }
        }
    ]

def display_server_health(servers: List[Dict[str, Any]]):
    """Display server health and status summary."""
    print("\n🏥 Server Health Summary:")
    print("=" * 50)
    
    total_servers = len(servers)
    healthy_servers = sum(1 for s in servers if "Success" in s["connection_test"])
    total_tools = sum(s["tool_count"] for s in servers)
    
    print(f"  📊 Total Servers: {total_servers}")
    print(f"  ✅ Healthy: {healthy_servers}/{total_servers}")
    print(f"  🔧 Total Tools: {total_tools}")
    
    if total_servers > 0:
        health_rate = (healthy_servers / total_servers) * 100
        if health_rate == 100:
            print(f"  🎉 Health Status: EXCELLENT ({health_rate:.0f}%)")
        elif health_rate >= 75:
            print(f"  👍 Health Status: GOOD ({health_rate:.0f}%)")
        elif health_rate >= 50:
            print(f"  ⚠️  Health Status: FAIR ({health_rate:.0f}%)")
        else:
            print(f"  🚨 Health Status: POOR ({health_rate:.0f}%)")

def display_capability_matrix(servers: List[Dict[str, Any]]):
    """Display a matrix of server capabilities."""
    print("\n📋 Server Capability Matrix:")
    print("=" * 50)
    
    if not servers:
        print("  No servers to analyze")
        return
    
    # Collect all capabilities
    all_capabilities = set()
    for server in servers:
        caps = server.get("capabilities", {})
        all_capabilities.update(caps.keys())
    
    # Header
    print(f"{'Server':<20} {'Tools':<6} {'Resources':<10} {'Prompts':<8} {'Streaming':<10} {'Notifications':<13}")
    print("-" * 75)
    
    for server in servers:
        name = server["name"][:19]
        caps = server.get("capabilities", {})
        features = server.get("features", {})
        
        tools_icon = "✅" if caps.get("tools") else "❌"
        resources_icon = "✅" if caps.get("resources") else "❌"
        prompts_icon = "✅" if caps.get("prompts") else "❌"
        streaming_icon = "✅" if features.get("streaming") else "❌"
        notifications_icon = "✅" if features.get("notifications") else "❌"
        
        print(f"{name:<20} {tools_icon:<6} {resources_icon:<10} {prompts_icon:<8} {streaming_icon:<10} {notifications_icon:<13}")

def display_performance_analysis(servers: List[Dict[str, Any]]):
    """Display performance analysis of servers."""
    print("\n⚡ Performance Analysis:")
    print("=" * 50)
    
    if not servers:
        print("  No performance data available")
        return
    
    print(f"{'Server':<20} {'Connection':<12} {'Ping Time':<12} {'Tool List':<12} {'Status'}")
    print("-" * 70)
    
    for server in servers:
        name = server["name"][:19]
        connection = server["connection_test"]
        perf = server.get("performance", {})
        
        ping_time = perf.get("ping_time", "N/A")
        tool_time = perf.get("tool_list_time", "N/A")
        
        # Status based on performance
        status = "🎭 Mock" if "Mock" in server["status"] else "Unknown"
        if "Success" in connection:
            # Analyze ping time if available
            if ping_time != "N/A" and "ms" in ping_time:
                ping_ms = float(ping_time.replace("ms", ""))
                if ping_ms < 10:
                    status = "🚀 Fast"
                elif ping_ms < 50:
                    status = "✅ Good"
                elif ping_ms < 100:
                    status = "⚠️ Slow"
                else:
                    status = "🐌 Very Slow"
            else:
                status = "✅ Connected"
        elif "Failed" in connection:
            status = "❌ Failed"
        elif "Timeout" in connection:
            status = "⏱️ Timeout"
        
        print(f"{name:<20} {connection[:11]:<12} {ping_time:<12} {tool_time:<12} {status}")

def display_protocol_compatibility(servers: List[Dict[str, Any]]):
    """Display protocol version compatibility analysis."""
    print("\n🔄 Protocol Compatibility:")
    print("=" * 50)
    
    if not servers:
        print("  No servers to analyze")
        return
    
    # Collect protocol versions
    versions = {}
    for server in servers:
        version = server.get("protocol_version", "unknown")
        if version not in versions:
            versions[version] = []
        versions[version].append(server["name"])
    
    print(f"  Protocol Versions Found: {len(versions)}")
    
    for version, server_names in versions.items():
        count = len(server_names)
        print(f"    {version}: {count} server(s) - {', '.join(server_names)}")
    
    # Check for compatibility issues
    if len(versions) > 1:
        print(f"\n  ⚠️  Multiple protocol versions detected!")
        print(f"     This may cause compatibility issues.")
        
        # Find newest and oldest
        version_list = [v for v in versions.keys() if v != "unknown"]
        if version_list:
            newest = max(version_list)
            oldest = min(version_list)
            print(f"     Newest: {newest}")
            print(f"     Oldest: {oldest}")
    else:
        print(f"\n  ✅ All servers use consistent protocol version")

def display_tool_inventory(servers: List[Dict[str, Any]]):
    """Display comprehensive tool inventory."""
    print("\n🔧 Tool Inventory:")
    print("=" * 50)
    
    if not servers:
        print("  No servers to analyze")
        return
    
    all_tools = {}
    for server in servers:
        server_name = server["name"]
        tools = server.get("tools", [])
        
        print(f"\n  📦 {server_name} ({len(tools)} tools):")
        if not tools:
            print("    No tools available")
        else:
            for tool in tools:
                tool_name = tool.get("name", "unknown")
                description = tool.get("description", "No description")
                print(f"    • {tool_name}: {description}")
                
                # Track for duplicate analysis
                if tool_name in all_tools:
                    all_tools[tool_name].append(server_name)
                else:
                    all_tools[tool_name] = [server_name]
    
    # Check for duplicate tool names
    duplicates = {name: servers for name, servers in all_tools.items() if len(servers) > 1}
    if duplicates:
        print(f"\n  ⚠️  Duplicate Tool Names Detected:")
        for tool_name, server_list in duplicates.items():
            print(f"    '{tool_name}' exists in: {', '.join(server_list)}")
        print(f"    This may cause namespace conflicts!")
    else:
        print(f"\n  ✅ No duplicate tool names detected")

async def run_diagnostics():
    """Run comprehensive MCP server diagnostics."""
    print("🔍 MCP Server Diagnostic Tool")
    print("=" * 60)
    print("This tool analyzes your MCP server environment and connections")
    print("=" * 60)
    
    # Server analysis (this will do environment check internally)
    servers = await analyze_real_servers()
    
    # Only continue with other analyses if we have real server data
    has_real_servers = any(not s.get("name", "").startswith("mock-") for s in servers)
    
    if not has_real_servers:
        print("\n⚠️  No real MCP servers found or analyzed.")
        print("To get real diagnostics:")
        print("1. Create an MCP config file (mcp_config.json or ~/.config/mcp/config.json)")
        print("2. Add server configurations with valid commands")
        print("3. Ensure server scripts are installed and executable")
        print("4. Re-run this diagnostic")
        print("\nShowing mock analysis for demonstration...")
    
    # Display results
    display_server_health(servers)
    display_capability_matrix(servers)
    display_performance_analysis(servers)
    display_protocol_compatibility(servers)
    display_tool_inventory(servers)
    
    # Enhanced recommendations based on real vs mock data
    print("\n💡 Recommendations:")
    print("=" * 50)
    
    if not has_real_servers:
        print("  🚨 No real MCP servers configured or working")
        print("  • Create a config file with MCP server definitions")
        print("  • Install MCP server packages (e.g., mcp-server-sqlite, mcp-server-filesystem)")
        print("  • Verify server commands are in your PATH")
        print("  • Check server script permissions and dependencies")
        print("\n  📝 Example config file (mcp_config.json):")
        print('     {')
        print('       "mcpServers": {')
        print('         "sqlite": {')
        print('           "command": "mcp-server-sqlite",')
        print('           "args": ["--db-path", "example.db"]')
        print('         },')
        print('         "filesystem": {')
        print('           "command": "mcp-server-filesystem",')
        print('           "args": ["--allowed-dir", "/path/to/directory"]')
        print('         }')
        print('       }')
        print('     }')
    else:
        healthy_count = sum(1 for s in servers if "Success" in s["connection_test"])
        if healthy_count == len(servers):
            print("  ✅ All servers are healthy - excellent!")
            print("  • Your MCP environment is working well")
            print("  • Consider exploring additional MCP servers for more functionality")
        elif healthy_count > 0:
            failed_servers = [s["name"] for s in servers if "Success" not in s["connection_test"]]
            print(f"  ⚠️  {len(failed_servers)} server(s) have issues: {', '.join(failed_servers)}")
            print("  • Check server installation and dependencies")
            print("  • Verify command paths and arguments in config")
            print("  • Review server logs for specific error messages")
        else:
            print("  🚨 No servers are responding properly")
            print("  • Verify all server commands are installed and executable")
            print("  • Check config file paths and arguments")
            print("  • Test server commands manually from command line")
        
        # Protocol recommendations
        versions = set(s.get("protocol_version", "unknown") for s in servers if not s.get("name", "").startswith("mock-"))
        if len(versions) > 1 and "unknown" not in versions:
            print("  • Consider updating servers to use the same protocol version")
        
        # Tool recommendations
        total_tools = sum(s["tool_count"] for s in servers)
        if total_tools == 0:
            print("  • Servers have no tools - check server implementations")
        elif total_tools < 5:
            print("  • Consider adding more MCP servers for additional functionality")
    
    server_count = len([s for s in servers if not s.get("name", "").startswith("mock-")])
    status = "real" if has_real_servers else "mock"
    print(f"\n🏁 Diagnostic complete - analyzed {server_count} {status} server(s)")
    
    if has_real_servers:
        print("✅ Real server analysis completed successfully!")
    else:
        print("⚠️  Run with real MCP servers for accurate diagnostics")

def main():
    """Main entry point."""
    try:
        asyncio.run(run_diagnostics())
    except KeyboardInterrupt:
        print("\n\n👋 Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()