#!/usr/bin/env python3
"""
Detailed test for Context7 integration with full result display
"""

import asyncio
import sys
import os
import json

# Add the current directory to the path
sys.path.insert(0, os.getcwd())


async def test_context7_detailed():
    """Test Context7 integration with detailed result inspection"""
    try:
        from mcp_cli.tools.manager import ToolManager
        from mcp_cli.cli_options import process_options

        print("🔧 Detailed Context7 Integration Test")
        print("=" * 50)

        # Setup
        servers, _, server_names = process_options(
            server="context7",
            disable_filesystem=False,
            provider="ollama",
            model="gpt-oss",
            config_file="server_config.json",
            quiet=False,
        )

        # Initialize ToolManager
        tm = ToolManager(
            config_file="server_config.json",
            servers=servers,
            server_names=server_names,
            tool_timeout=60.0,
        )

        success = await tm.initialize()
        if not success:
            print("❌ Failed to initialize ToolManager")
            return False

        print("✅ ToolManager initialized")

        # Get tools
        print("\n🔍 Discovering tools...")
        tools = await tm.get_all_tools()

        print(f"📊 Found {len(tools)} tools:")
        for tool in tools:
            print(f"  🔧 {tool.namespace}.{tool.name}")
            print(f"     📝 {tool.description}")
            if tool.parameters:
                params = tool.parameters.get("properties", {})
                print(f"     🎛️  Parameters: {list(params.keys())}")

        # Test resolve-library-id with detailed output
        print("\n🧪 Testing resolve-library-id tool...")
        print("=" * 30)

        result = await tm.execute_tool(
            "context7.resolve-library-id", {"libraryName": "react"}
        )

        print(f"Success: {result.success}")
        print(f"Error: {result.error}")
        print(f"Execution time: {result.execution_time}")
        print(f"Result type: {type(result.result)}")
        print(f"Result length: {len(str(result.result)) if result.result else 'None'}")

        if result.success and result.result:
            print("\n📤 Full Result:")
            print("-" * 40)

            # Handle different result types
            if isinstance(result.result, str):
                print(result.result)
            elif isinstance(result.result, (dict, list)):
                print(json.dumps(result.result, indent=2))
            else:
                print(str(result.result))

            print("-" * 40)
        else:
            print("❌ Tool execution failed or returned empty result")

        # Test get-library-docs if resolve worked
        if result.success and result.result and isinstance(result.result, str):
            print("\n🧪 Testing get-library-docs tool...")
            print("=" * 30)

            # Try to extract a library ID from the result
            lines = result.result.split("\n")
            library_id = None

            # Look for patterns like "/facebook/react" in the result
            for line in lines:
                if (
                    "/" in line
                    and not line.startswith("Each")
                    and not line.startswith("-")
                ):
                    # Extract potential library ID
                    parts = line.split()
                    for part in parts:
                        if part.startswith("/") and "/" in part[1:]:
                            library_id = part
                            break
                    if library_id:
                        break

            if library_id:
                print(f"📚 Testing documentation retrieval for: {library_id}")

                docs_result = await tm.execute_tool(
                    "context7.get-library-docs",
                    {
                        "context7CompatibleLibraryID": library_id,
                        "topic": "getting started",
                        "tokens": 5000,
                    },
                )

                print(f"Success: {docs_result.success}")
                print(f"Error: {docs_result.error}")
                print(f"Execution time: {docs_result.execution_time}")

                if docs_result.success and docs_result.result:
                    docs_content = str(docs_result.result)
                    print(f"📄 Documentation length: {len(docs_content)} characters")
                    print("📄 First 200 characters:")
                    print(
                        docs_content[:200] + "..."
                        if len(docs_content) > 200
                        else docs_content
                    )
                else:
                    print("❌ Documentation retrieval failed")
            else:
                print("⚠️  Could not extract library ID from resolve result")

        # Test LLM integration
        print("\n🤖 Testing LLM Integration...")
        print("=" * 30)

        llm_tools, name_mapping = await tm.get_adapted_tools_for_llm("openai")

        print(f"📋 LLM Tools Generated: {len(llm_tools)}")
        print(f"🔗 Name Mapping Entries: {len(name_mapping)}")

        for i, tool in enumerate(llm_tools):
            func = tool["function"]
            print(f"\n🔧 Tool {i + 1}: {func['name']}")
            print(f"   📝 Description: {func['description'][:80]}...")
            print(f"   🔗 Maps to: {name_mapping.get(func['name'], 'Unknown')}")

            # Show parameters
            params = func.get("parameters", {}).get("properties", {})
            if params:
                print(f"   🎛️  Parameters: {list(params.keys())}")

        # Cleanup
        await tm.close()
        print("\n✅ Detailed test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_context7_detailed())
    print(f"\n{'🎉 SUCCESS' if success else '💥 FAILED'}")
    sys.exit(0 if success else 1)
