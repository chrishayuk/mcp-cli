#!/usr/bin/env python3
"""
MCP Server Integration Diagnostic

Tests MCP server integration with the tool system.
Can be used with any MCP server configured in server_config.json.
"""

import asyncio
import json
import logging
import sys
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose logs
logging.getLogger("chuk_tool_processor").setLevel(logging.WARNING)
logging.getLogger("chuk_mcp").setLevel(logging.WARNING)


async def test_server_integration(server_name: Optional[str] = None):
    """Test MCP server integration."""

    print("\n🔧 MCP Server Integration Test")
    print("=" * 50)

    # Import required components
    from mcp_cli.tools.manager import ToolManager

    # Load server config to find available servers
    config_file = "server_config.json"
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
            available_servers = list(config.get("mcpServers", {}).keys())
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

    if not available_servers:
        print("❌ No servers configured in server_config.json")
        return False

    # Use provided server or first available
    if not server_name:
        server_name = available_servers[0]
        print(f"ℹ️  No server specified, using: {server_name}")
    elif server_name not in available_servers:
        print(f"❌ Server '{server_name}' not found")
        print(f"   Available servers: {', '.join(available_servers)}")
        return False

    print(f"📡 Testing server: {server_name}")

    # Initialize ToolManager
    print("✅ Initializing ToolManager...", end="")
    tool_manager = ToolManager(config_file, [server_name])
    if not await tool_manager.initialize():
        print(" ❌")
        return False
    print(" ✓")

    # Try to get protocol version
    protocol_version = "unknown"
    try:
        if hasattr(tool_manager, "stream_manager") and hasattr(
            tool_manager.stream_manager, "streams"
        ):
            if len(tool_manager.stream_manager.streams) > 0:
                stream = tool_manager.stream_manager.streams[0]
                if hasattr(stream, "protocol_version"):
                    protocol_version = stream.protocol_version
                elif hasattr(stream, "_protocol_version"):
                    protocol_version = stream._protocol_version
                elif hasattr(stream, "client") and hasattr(
                    stream.client, "protocol_version"
                ):
                    protocol_version = stream.client.protocol_version

        if protocol_version != "unknown":
            print(f"📋 Protocol version: {protocol_version}")
    except:
        pass

    # Discover tools
    print("\n🔍 Discovering tools...")
    tools = await tool_manager.get_all_tools()
    print(f"📊 Found {len(tools)} tools:")

    for tool in tools[:5]:  # Show first 5 tools
        # Handle both dict and object formats
        if hasattr(tool, "name"):
            name = tool.name
            desc = getattr(tool, "description", "No description")
        else:
            name = tool.get("name", "Unknown")
            desc = tool.get("description", "No description")
        print(f"  • {name}: {desc[:50]}...")

    if len(tools) > 5:
        print(f"  ... and {len(tools) - 5} more")

    # Test a simple tool if available
    if tools:
        test_tool = tools[0]

        # Handle both dict and object formats
        if hasattr(test_tool, "name"):
            tool_name = test_tool.name
            schema = getattr(test_tool, "inputSchema", {})
        else:
            tool_name = test_tool.get("name", "Unknown")
            schema = test_tool.get("inputSchema", {})

        print(f"\n🧪 Testing tool: {tool_name}")
        print("=" * 30)

        # Build arguments based on schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Try to build minimal valid arguments
        test_args = {}
        for prop_name in required[:1]:  # Just test first required param
            prop_schema = properties.get(prop_name, {})
            if prop_schema.get("type") == "string":
                test_args[prop_name] = "test"
            elif prop_schema.get("type") == "number":
                test_args[prop_name] = 1
            elif prop_schema.get("type") == "boolean":
                test_args[prop_name] = True

        try:
            result = await tool_manager.execute_tool(tool_name, test_args)
            if result.success:
                print(f"✅ Success: {str(result.result)[:100]}...")
            else:
                print(f"⚠️  Error: {result.error}")
        except Exception as e:
            print(f"❌ Exception: {e}")

    # Cleanup
    await tool_manager.close()

    print("\n✅ Test completed successfully!")
    return True


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Server Integration Test")
    parser.add_argument(
        "--server", help="Server name from server_config.json (optional)"
    )

    args = parser.parse_args()

    try:
        success = await test_server_integration(args.server)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
