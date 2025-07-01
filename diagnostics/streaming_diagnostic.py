#!/usr/bin/env python3
# diagnostics/streaming_diagnostic.py - Test chuk-llm streaming directly

import asyncio
import os
import sys
import json
import time
import logging
from pathlib import Path

# Add the parent directory to the path so we can import mcp_cli modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from chuk_llm.llm.client import get_client
from mcp_cli.model_manager import ModelManager
from mcp_cli.chat.streaming_handler import StreamingResponseHandler
from mcp_cli.logging_config import setup_logging

# Set up minimal logging to avoid spam
setup_logging(level="WARNING")

async def test_chuk_llm_client_directly():
    """Test chuk-llm client capabilities directly"""
    print("=" * 60)
    print("TESTING CHUK-LLM CLIENT DIRECTLY")
    print("=" * 60)
    
    try:
        # Test with get_client (the newer interface)
        print("\n1. Testing get_client() interface...")
        client = get_client("openai", model="gpt-4o-mini")
        print(f"✓ Client created: {type(client)}")
        
        # Inspect client methods
        methods = [m for m in dir(client) if not m.startswith('_') and callable(getattr(client, m))]
        print(f"Available methods: {methods}")
        
        # Check for streaming support
        has_create_completion = hasattr(client, 'create_completion')
        has_stream_completion = hasattr(client, 'stream_completion')
        print(f"Has create_completion: {has_create_completion}")
        print(f"Has stream_completion: {has_stream_completion}")
        
        if has_create_completion:
            import inspect
            try:
                sig = inspect.signature(client.create_completion)
                params = list(sig.parameters.keys())
                has_stream_param = 'stream' in params
                print(f"create_completion parameters: {params}")
                print(f"Supports stream parameter: {has_stream_param}")
            except Exception as e:
                print(f"Could not inspect create_completion: {e}")
        
        return client
        
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        return None

async def test_regular_completion(client):
    """Test regular (non-streaming) completion"""
    print("\n" + "=" * 60)
    print("TESTING REGULAR COMPLETION")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "Write a very short haiku about programming. Just 3 lines."}
    ]
    
    try:
        print("Making regular completion request...")
        start_time = time.time()
        
        response = await client.create_completion(messages)
        elapsed = time.time() - start_time
        
        print(f"✓ Regular completion successful in {elapsed:.2f}s")
        print(f"Response type: {type(response)}")
        print(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
        
        if isinstance(response, dict):
            content = response.get("response") or response.get("content") or response.get("message", "")
            print(f"Content: {content}")
        else:
            print(f"Raw response: {response}")
            
        return True
        
    except Exception as e:
        print(f"✗ Regular completion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_streaming_completion(client, quiet_mode=True):
    """Test streaming completion"""
    print("\n" + "=" * 60)
    print("TESTING STREAMING COMPLETION")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "Write a short story about a robot learning to paint. Keep it brief but creative."}
    ]
    
    # Test 1: Direct streaming with create_completion
    print("\n--- Test 1: Direct streaming with create_completion(stream=True) ---")
    try:
        print("Attempting streaming with stream=True...")
        
        full_response = ""
        chunk_count = 0
        start_time = time.time()
        
        async for chunk in client.create_completion(messages, stream=True):
            chunk_count += 1
            elapsed = time.time() - start_time
            
            # Only show chunk info if not in quiet mode
            if not quiet_mode and (chunk_count <= 3 or chunk_count % 50 == 0):
                print(f"Chunk {chunk_count} (t={elapsed:.2f}s): {type(chunk)} - {str(chunk)[:80]}...")
            
            # Extract content from chunk (multiple possible formats)
            content = ""
            if isinstance(chunk, dict):
                content = (chunk.get("response") or 
                          chunk.get("content") or 
                          chunk.get("text") or "")
                
                if "delta" in chunk:
                    delta = chunk["delta"]
                    if isinstance(delta, dict):
                        content = delta.get("content", "")
                        
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        content = choice["delta"]["content"] or ""
            elif isinstance(chunk, str):
                content = chunk
            
            if content:
                print(content, end="", flush=True)
                full_response += content
                
        print(f"\n✓ Streaming completed! Chunks: {chunk_count}, Total time: {time.time() - start_time:.2f}s")
        print(f"Full response length: {len(full_response)} chars")
        
        return True
        
    except TypeError as e:
        print(f"✗ Streaming failed - method signature issue: {e}")
        return False
    except Exception as e:
        print(f"✗ Streaming failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mcp_streaming_handler(client):
    """Test the MCP CLI streaming handler"""
    print("\n" + "=" * 60)
    print("TESTING MCP CLI STREAMING HANDLER")
    print("=" * 60)
    
    try:
        print("Creating StreamingResponseHandler...")
        handler = StreamingResponseHandler()
        
        messages = [
            {"role": "user", "content": "Explain how a computer works in simple terms. Be detailed but clear."}
        ]
        
        print("Testing StreamingResponseHandler.stream_response()...")
        start_time = time.time()
        
        # Reduce verbosity during streaming test
        print("Starting streaming test (output suppressed for readability)...")
        
        result = await handler.stream_response(
            client=client,
            messages=messages
        )
        
        elapsed = time.time() - start_time
        print(f"✓ StreamingResponseHandler completed in {elapsed:.2f}s")
        
        print(f"Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
            print(f"Response length: {len(result.get('response', ''))}")
            print(f"Chunks received: {result.get('chunks_received', 0)}")
            print(f"Was streaming: {result.get('streaming', False)}")
            print(f"Tool calls: {len(result.get('tool_calls', []))}")
            
            response_content = result.get('response', '')
            if response_content:
                print(f"Response preview: {response_content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ MCP streaming handler failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_manager_integration():
    """Test ModelManager integration"""
    print("\n" + "=" * 60)
    print("TESTING MODEL MANAGER INTEGRATION")
    print("=" * 60)
    
    try:
        print("Creating ModelManager...")
        model_manager = ModelManager()
        
        print(f"Active provider: {model_manager.get_active_provider()}")
        print(f"Active model: {model_manager.get_active_model()}")
        
        print("Getting client through ModelManager...")
        client = model_manager.get_client()
        print(f"✓ Client obtained: {type(client)}")
        
        # Test a simple completion through model manager
        print("Testing completion through ModelManager client...")
        messages = [{"role": "user", "content": "Say hello in exactly 5 words."}]
        
        response = await client.create_completion(messages)
        print(f"✓ ModelManager client works: {response}")
        
        return client
        
    except Exception as e:
        print(f"✗ ModelManager integration failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_conversation_flow():
    """Test the conversation flow like the actual chat does"""
    print("\n" + "=" * 60)
    print("TESTING CONVERSATION FLOW")
    print("=" * 60)
    
    try:
        # Simulate the chat context initialization
        model_manager = ModelManager()
        client = model_manager.get_client()
        
        conversation_history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a very short joke about programming."}
        ]
        
        # Test regular flow
        print("Testing regular conversation flow...")
        response = await client.create_completion(messages=conversation_history)
        print(f"✓ Regular flow: {response}")
        
        # Test streaming flow
        print("\nTesting streaming conversation flow...")
        handler = StreamingResponseHandler()
        
        streaming_result = await handler.stream_response(
            client=client,
            messages=conversation_history
        )
        
        print(f"✓ Streaming flow completed")
        print(f"Response: {streaming_result.get('response', '')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Conversation flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all streaming diagnostic tests"""
    print("CHUK-LLM STREAMING DIAGNOSTIC")
    print("Testing streaming functionality for MCP CLI")
    print("=" * 60)
    
    # Silence noisy loggers
    logging.getLogger("markdown_it").setLevel(logging.WARNING)
    logging.getLogger("chuk_llm").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Test 1: Direct client creation and capabilities
    client = await test_chuk_llm_client_directly()
    if not client:
        print("\n❌ Cannot proceed - failed to create client")
        return
    
    # Test 2: Regular completion
    regular_works = await test_regular_completion(client)
    if not regular_works:
        print("\n⚠️  Regular completion failed - this is a serious issue")
    
    # Test 3: Streaming completion (quiet mode to reduce output)
    streaming_works = await test_streaming_completion(client, quiet_mode=True)
    
    # Test 4: MCP streaming handler
    handler_works = await test_mcp_streaming_handler(client)
    
    # Test 5: ModelManager integration
    model_manager_client = await test_model_manager_integration()
    
    # Test 6: Conversation flow
    conversation_works = await test_conversation_flow()
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    results = {
        "Regular completion": regular_works,
        "Direct streaming": streaming_works,
        "MCP streaming handler": handler_works,
        "ModelManager integration": model_manager_client is not None,
        "Conversation flow": conversation_works,
    }
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if not regular_works:
        print("🔴 CRITICAL: Regular completion is broken - check chuk-llm installation")
    elif not streaming_works:
        print("🟡 WARNING: Streaming not working - MCP CLI will fall back to regular completion")
    elif not handler_works:
        print("🟡 WARNING: MCP streaming handler has issues - check implementation")
    else:
        print("🟢 ALL GOOD: Streaming should work properly in MCP CLI")
    
    # Configuration info
    print(f"\nEnvironment info:")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check for API keys
    api_keys = {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
    }
    
    print(f"API keys configured:")
    for key, present in api_keys.items():
        status = "✓" if present else "✗"
        print(f"  {key}: {status}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
    except Exception as e:
        print(f"\n\nDiagnostic failed with error: {e}")
        import traceback
        traceback.print_exc()