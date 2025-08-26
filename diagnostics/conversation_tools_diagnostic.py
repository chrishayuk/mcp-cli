#!/usr/bin/env python3
"""
CLI Conversation Flow Diagnostic (Simplified)

Tests the conversation flow that happens when using the CLI with tool calls.
"""

import asyncio
import logging
import sys
from typing import List

# Configure logging with reduced verbosity
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Suppress debug logs from other modules
logging.getLogger("mcp_cli").setLevel(logging.WARNING)
logging.getLogger("chuk_mcp").setLevel(logging.WARNING)
logging.getLogger("chuk_llm").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chuk_tool_processor").setLevel(logging.WARNING)
logging.getLogger("chuk_sessions").setLevel(logging.WARNING)
logging.getLogger("chuk_ai_session_manager").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)


async def test_conversation_flow(config_file: str, servers: List[str]) -> bool:
    """Test the conversation flow with minimal output."""

    print("\n🎯 Testing Conversation Flow")
    print("-" * 40)

    # Import required components
    from mcp_cli.tools.manager import ToolManager
    from mcp_cli.chat.chat_context import ChatContext

    # Initialize ToolManager
    print("1. Initializing ToolManager...", end="")
    tool_manager = ToolManager(config_file, servers)
    if not await tool_manager.initialize():
        print(" ❌")
        return False
    print(" ✅")

    # Create ChatContext
    print("2. Creating ChatContext...", end="")
    chat_context = ChatContext.create(
        tool_manager=tool_manager, provider="ollama", model="gpt-oss"
    )

    if not await chat_context.initialize():
        print(" ❌")
        await tool_manager.close()
        return False
    print(" ✅")

    # Test tool discovery
    print(f"3. Tools discovered: {len(chat_context.tools)} tools")

    # Test a simple tool call
    print("4. Testing tool execution...", end="")
    try:
        result = await tool_manager.execute_tool("list_tables", {})
        if result.success:
            print(" ✅")
            print(f"   Result: {result.result[:100]}...")
        else:
            print(f" ❌ {result.error}")
    except Exception as e:
        print(f" ❌ {e}")

    # Test tool call with arguments
    print("5. Testing tool with arguments...", end="")
    try:
        result = await tool_manager.execute_tool("read_query", {"query": "SELECT 1"})
        if result.success:
            print(" ✅")
        else:
            print(f" ❌ {result.error}")
    except Exception as e:
        print(f" ❌ {e}")

    # Cleanup
    await tool_manager.close()

    print("-" * 40)
    print("✅ Conversation flow test completed\n")
    return True


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Conversation Flow Test")
    parser.add_argument(
        "--config", default="server_config.json", help="Server config file"
    )
    parser.add_argument(
        "--server", action="append", default=[], help="Server names to test"
    )

    args = parser.parse_args()

    if not args.server:
        args.server = ["sqlite"]

    try:
        success = await test_conversation_flow(args.config, args.server)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
