#!/usr/bin/env python
"""
Test module for LLM client functionality.
This file can be run directly to test the LLM client with:
python -m mcp_cli.llm
"""
import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Any

# Import LLM-related functionality
from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator
from mcp_cli.llm.tools_handler import convert_to_openai_tools


async def test_llm_client(provider: str = "openai",
                        model: str = "gpt-4o",
                        prompt: str = "Hello, how are you?",
                        tools: List[Dict[str, Any]] = None,
                        verbose: bool = False):
    """Test the LLM client with a simple prompt."""
    print(f"\n=== Testing LLM Client ===")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Prompt: '{prompt}'")
    
    # Create the client
    try:
        client = get_llm_client(provider=provider, model=model)
        print(f"Client created successfully: {type(client).__name__}")
    except Exception as e:
        print(f"Error creating client: {e}")
        return False

    # Generate a system prompt
    prompt_generator = SystemPromptGenerator()
    if tools:
        tools_dict = {"tools": tools}
        system_prompt = prompt_generator.generate_prompt(tools_dict)
        openai_tools = convert_to_openai_tools(tools)
        print(f"Generated system prompt with {len(tools)} tools")
        if verbose:
            print(f"System prompt: {system_prompt[:300]}...\n")
            print(f"Tools: {json.dumps(tools, indent=2)}")
    else:
        system_prompt = prompt_generator.generate_prompt({})
        openai_tools = None
        print("Generated default system prompt without tools")
    
    # Create messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Send the completion request
    try:
        print("\nSending request to LLM...")
        start_time = asyncio.get_event_loop().time()
        
        if tools:
            completion = client.create_completion(messages=messages, tools=openai_tools)
        else:
            completion = client.create_completion(messages=messages)
            
        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time
        
        print(f"Request completed in {elapsed:.2f} seconds")
        
        # Check completion structure
        print("\n== Completion Structure ==")
        if completion is None:
            print("Error: Completion is None")
            return False
            
        print(f"Completion type: {type(completion).__name__}")
        print(f"Completion keys: {list(completion.keys())}")
        
        # Check for tool calls
        if "tool_calls" in completion and completion["tool_calls"]:
            print(f"\nTool calls requested: {len(completion['tool_calls'])}")
            for i, tool_call in enumerate(completion["tool_calls"]):
                if isinstance(tool_call, dict) and "function" in tool_call:
                    fn = tool_call["function"]
                    print(f"  Tool {i+1}: {fn.get('name', 'unknown')}")
                    print(f"    Arguments: {fn.get('arguments', 'none')}")
                else:
                    print(f"  Tool {i+1}: {tool_call}")
            
            # Response should be None with tool calls
            if "response" in completion:
                print(f"\nResponse with tool calls: {completion.get('response')}")
        
        # Check for direct response
        if "response" in completion and completion["response"]:
            print(f"\nDirect response: {completion['response'][:150]}...")
        elif "response" in completion and completion["response"] is None:
            print("\nResponse key exists but is None")
        else:
            print("\nNo 'response' key in completion")
            
        # Print the full completion in verbose mode
        if verbose:
            print("\n== Full Completion ==")
            print(json.dumps(completion, indent=2, default=str))
            
        return True
        
    except Exception as e:
        print(f"Error during completion: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def setup_mock_tools() -> List[Dict[str, Any]]:
    """Create mock tools for testing."""
    return [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "search_web",
            "description": "Search the web for information",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    ]


async def main():
    """Run the main test function."""
    parser = argparse.ArgumentParser(description='Test the LLM client')
    parser.add_argument('--provider', type=str, default="openai", help='LLM provider (openai, ollama)')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Model name')
    parser.add_argument('--prompt', type=str, default="Tell me about the weather in New York.", 
                      help='Prompt to send to the model')
    parser.add_argument('--tools', action='store_true', help='Include mock tools')
    parser.add_argument('--verbose', action='store_true', help='Show verbose output')
    
    args = parser.parse_args()
    
    # Check for OpenAI API key if using OpenAI
    if args.provider.lower() == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set. OpenAI client will run in mock mode.")
        
    # Set up mock tools if requested
    tools = setup_mock_tools() if args.tools else None
    
    # Run the test
    success = await test_llm_client(
        provider=args.provider,
        model=args.model,
        prompt=args.prompt,
        tools=tools,
        verbose=args.verbose
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())