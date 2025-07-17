#!/usr/bin/env python3
"""
Focused Streaming Diagnostic - Minimal Test

This strips away all complexity and focuses purely on:
1. Does the LLM send proper parameters in streaming chunks?
2. Is the streaming handler processing them correctly?
3. Where exactly does the data get lost?

No conversation flows, no complex prompts - just direct tool calling.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path and load environment
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment
try:
    from dotenv import load_dotenv
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded .env from {env_file}")
    else:
        load_dotenv()
        print("✅ Loaded .env from system")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment")

# Setup basic logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_simple_streaming_tool_call():
    """Test the simplest possible streaming tool call that should have parameters."""
    
    print("🔍 FOCUSED STREAMING DIAGNOSTIC")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        return False
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="openai", model="gpt-4o-mini")
        print(f"✅ Client: {type(client).__name__}")
        
        # Single tool that MUST have parameters
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute a SQL query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        # Super explicit message that should force parameters
        messages = [
            {
                "role": "user",
                "content": "Call execute_sql with the query 'SELECT * FROM users LIMIT 5'"
            }
        ]
        
        print("\n🎯 TEST: Explicit SQL tool call")
        print("Request: Call execute_sql with query 'SELECT * FROM users LIMIT 5'")
        print("Expected: Tool call with proper query parameter")
        
        # Test streaming
        print("\n🌊 Testing with STREAMING:")
        streaming_result = await test_streaming_call(client, messages, tools)
        
        # Test non-streaming for comparison
        print("\n📝 Testing WITHOUT streaming:")
        regular_result = await test_regular_call(client, messages, tools)
        
        # Compare results
        print("\n📊 COMPARISON:")
        print(f"Streaming tool calls: {len(streaming_result)}")
        print(f"Regular tool calls: {len(regular_result)}")
        
        if streaming_result and regular_result:
            stream_args = streaming_result[0].get("function", {}).get("arguments", "")
            regular_args = regular_result[0].get("function", {}).get("arguments", "")
            
            print(f"Streaming args: '{stream_args}'")
            print(f"Regular args: '{regular_args}'")
            
            if stream_args == regular_args and stream_args:
                print("✅ STREAMING WORKS - Results identical!")
                return True
            elif not stream_args and regular_args:
                print("❌ STREAMING BROKEN - Loses parameters!")
                return False
            else:
                print("⚠️  Both methods have issues")
                return False
        else:
            print("❌ No tool calls received")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming_call(client, messages, tools):
    """Test streaming and capture all chunks."""
    print("  Starting streaming...")
    
    all_chunks = []
    tool_calls = []
    
    try:
        async for chunk in client.create_completion(
            messages=messages,
            tools=tools,
            stream=True
        ):
            all_chunks.append(chunk)
            
            # Log each chunk
            chunk_num = len(all_chunks)
            print(f"    Chunk {chunk_num}: {json.dumps(chunk)}")
            
            # Extract tool calls
            if chunk.get("tool_calls"):
                for tc in chunk["tool_calls"]:
                    # Check if this is a new tool call or update to existing
                    tc_id = tc.get("id", f"unknown_{len(tool_calls)}")
                    
                    # Find existing or create new
                    existing = None
                    for existing_tc in tool_calls:
                        if existing_tc.get("id") == tc_id:
                            existing = existing_tc
                            break
                    
                    if existing is None:
                        # New tool call
                        new_tc = {
                            "id": tc_id,
                            "type": tc.get("type", "function"),
                            "function": {
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": tc.get("function", {}).get("arguments", "")
                            }
                        }
                        tool_calls.append(new_tc)
                        print(f"      ➕ New tool call: {new_tc}")
                    else:
                        # Update existing
                        if "function" in tc:
                            func = tc["function"]
                            if "name" in func:
                                existing["function"]["name"] += str(func["name"])
                            if "arguments" in func:
                                existing["function"]["arguments"] += str(func["arguments"])
                        print(f"      🔄 Updated tool call: {existing}")
    
    except Exception as e:
        print(f"    ❌ Streaming error: {e}")
    
    print(f"  ✅ Streaming complete: {len(all_chunks)} chunks, {len(tool_calls)} tool calls")
    
    # Log final tool calls
    for i, tc in enumerate(tool_calls):
        print(f"    Tool {i+1}: {tc['function']['name']}({tc['function']['arguments']})")
    
    return tool_calls


async def test_regular_call(client, messages, tools):
    """Test regular non-streaming call."""
    print("  Starting regular call...")
    
    try:
        result = await client.create_completion(
            messages=messages,
            tools=tools,
            stream=False
        )
        
        tool_calls = result.get("tool_calls", [])
        print(f"  ✅ Regular complete: {len(tool_calls)} tool calls")
        
        # Log tool calls
        for i, tc in enumerate(tool_calls):
            print(f"    Tool {i+1}: {tc['function']['name']}({tc['function']['arguments']})")
        
        return tool_calls
        
    except Exception as e:
        print(f"    ❌ Regular call error: {e}")
        return []


async def test_even_simpler():
    """Test with the absolute simplest possible case."""
    print("\n🔍 EVEN SIMPLER TEST")
    print("=" * 25)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        from chuk_llm.llm.client import get_client
        client = get_client(provider="openai", model="gpt-4o-mini")
        
        # Minimal tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "say_hello",
                    "description": "Say hello to someone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name to greet"}
                        },
                        "required": ["name"]
                    }
                }
            }
        ]
        
        # Ultra explicit
        messages = [
            {"role": "user", "content": "Call say_hello with name 'Alice'"}
        ]
        
        print("Request: Call say_hello with name 'Alice'")
        
        # Test both
        print("\n🌊 Streaming:")
        stream_result = await test_streaming_call(client, messages, tools)
        
        print("\n📝 Regular:")
        regular_result = await test_regular_call(client, messages, tools)
        
        # Check if they match
        if stream_result and regular_result:
            stream_name = json.loads(stream_result[0]["function"]["arguments"]).get("name", "")
            regular_name = json.loads(regular_result[0]["function"]["arguments"]).get("name", "")
            
            print(f"\nStreaming extracted name: '{stream_name}'")
            print(f"Regular extracted name: '{regular_name}'")
            
            if stream_name == regular_name == "Alice":
                print("✅ Both methods work perfectly!")
                return True
            else:
                print("❌ Methods differ or missing data!")
                return False
        else:
            print("❌ One or both methods failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run focused diagnostics."""
    print("🚀 FOCUSED STREAMING DIAGNOSTIC")
    print("Testing ONLY streaming vs non-streaming parameter extraction")
    
    # Test 1: Simple SQL
    result1 = await test_simple_streaming_tool_call()
    
    # Test 2: Even simpler
    result2 = await test_even_simpler()
    
    print("\n" + "=" * 50)
    print("🎯 FOCUSED DIAGNOSTIC RESULTS:")
    print(f"SQL Test: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"Simple Test: {'✅ PASS' if result2 else '❌ FAIL'}")
    
    if result1 and result2:
        print("\n✅ STREAMING WORKS PERFECTLY!")
        print("The issue is likely elsewhere in the MCP CLI pipeline.")
    elif result1 or result2:
        print("\n⚠️  MIXED RESULTS")
        print("Streaming works sometimes - need to investigate specific cases.")
    else:
        print("\n❌ STREAMING IS BROKEN")
        print("Parameters are being lost in the streaming process.")


if __name__ == "__main__":
    asyncio.run(main())