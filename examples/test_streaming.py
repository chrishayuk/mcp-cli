#!/usr/bin/env python3
# test_streaming.py - Test chuk-llm streaming directly

import asyncio
import os
from chuk_llm.llm.llm_client import get_llm_client

async def test_streaming():
    """Test if chuk-llm streaming works directly"""
    
    print("Testing chuk-llm streaming...")
    
    # Get the same client that MCP CLI would use
    client = get_llm_client("openai", model="gpt-4o-mini")
    
    print(f"Client type: {type(client)}")
    print(f"Client methods: {[m for m in dir(client) if not m.startswith('_')]}")
    
    # Test if create_completion has stream parameter
    import inspect
    try:
        sig = inspect.signature(client.create_completion)
        print(f"create_completion signature: {sig}")
        params = list(sig.parameters.keys())
        print(f"Parameters: {params}")
        has_stream = 'stream' in params
        print(f"Has 'stream' parameter: {has_stream}")
    except Exception as e:
        print(f"Could not inspect signature: {e}")
    
    messages = [
        {"role": "user", "content": "Write a short haiku about programming."}
    ]
    
    print("\n--- Testing regular completion ---")
    try:
        response = await client.create_completion(messages)
        print(f"Regular response: {response}")
    except Exception as e:
        print(f"Regular completion failed: {e}")
    
    print("\n--- Testing streaming completion ---")
    try:
        print("Attempting streaming with stream=True...")
        full_response = ""
        chunk_count = 0
        
        async for chunk in client.create_completion(messages, stream=True):
            chunk_count += 1
            print(f"Chunk {chunk_count}: {chunk}")
            
            if chunk.get("response"):
                content = chunk["response"]
                print(content, end="", flush=True)
                full_response += content
                
        print(f"\nStreaming completed. Total chunks: {chunk_count}")
        print(f"Full response: {full_response}")
        
    except TypeError as e:
        print(f"Streaming failed - method doesn't accept stream parameter: {e}")
    except Exception as e:
        print(f"Streaming failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_streaming())