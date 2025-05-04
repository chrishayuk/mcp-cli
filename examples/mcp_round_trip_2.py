#!/usr/bin/env python
"""
MCP "round-trip" example with SQLite stdio transport.

1. Starts the MCP-SQLite server via stdio
2. Discovers all sqlite.* tools and prints them
3. Builds OpenAI-compatible tool definitions for MCP tools
4. Sends your prompt to the LLM
5. Processes the assistant's response with the tool processor
6. Prints the function results
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

# chuk tool processor
from chuk_tool_processor.mcp import setup_mcp_stdio
from chuk_tool_processor.registry import ToolRegistryProvider

# mcp cli
from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator

# Import the tool conversion helper from MCP CLI
from mcp_cli.llm.tools_handler import convert_to_openai_tools

async def main() -> None:
    # load the environment variables
    load_dotenv()

    # parse arguments
    p = argparse.ArgumentParser(description="MCP round-trip over SQLite stdio")
    p.add_argument("--provider", default="openai", help="LLM provider (openai|ollama)")
    p.add_argument("--model",    default="gpt-4o-mini", help="Model name")
    p.add_argument("--prompt",   required=True, help="User prompt for the LLM")
    args = p.parse_args()

    # 1) Spin up stdio transport speaking to your sqlite MCP server
    processor, stream_mgr = await setup_mcp_stdio(
        config_file="server_config.json",
        servers=["sqlite"],
        server_names={0: "sqlite"},
        namespace="sqlite",
    )

    # 2) List & print all sqlite.* tools
    registry = ToolRegistryProvider.get_registry()
    sqlite_tools = [(ns, name) for ns, name in registry.list_tools() if ns == "sqlite"]
    print("🔧  Registered MCP tools:")
    for ns, name in sqlite_tools:
        md = registry.get_metadata(name, ns)
        print(f"  • {ns}.{name:<20} – {md.description or '<no description>'}")
    print()

    # 3) Build the OpenAI-functions list properly for MCP tools
    # First, build tool definitions with modified names (replace dots with underscores)
    mcp_tools = []
    original_to_openai_name = {}
    
    # loop through the tools
    for ns, name in sqlite_tools:
        md = registry.get_metadata(name, ns)
        if md:
            # Create OpenAI-compatible name (replace dots with underscores)
            openai_name = f"{ns}_{name}"
            original_name = f"{ns}.{name}"
            original_to_openai_name[openai_name] = original_name
            
            mcp_tools.append({
                "name": openai_name,  # OpenAI-compatible name
                "description": f"{md.description or ''} (Original tool: {original_name})",
                "parameters": md.argument_schema or {}
            })
    
    # Convert to OpenAI format
    funcs = convert_to_openai_tools(mcp_tools)
    
    # 4) Send prompt to LLM with our functions
    client = get_llm_client(provider=args.provider, model=args.model)
    sys_prompt = SystemPromptGenerator().generate_prompt({"tools": funcs})
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": args.prompt},
    ]
    completion = await client.create_completion(messages=messages, tools=funcs)
    
    # Print the assistant's response
    reply = completion.get("response", "")
    tool_calls = completion.get("tool_calls", [])
    
    if reply:
        print("=== Assistant Reply ===")
        print(reply, end="\n\n")
    
    # 5) Process any tool calls
    if tool_calls:
        print("=== Tool Calls ===")
        
        # Create a formatted response with tool calls in the MCP format
        mcp_text = ""
        for tc in tool_calls:
            if tc.get("function") and "name" in tc.get("function", {}):
                openai_name = tc["function"]["name"]
                # Convert back to original name format for processing
                if openai_name in original_to_openai_name:
                    original_name = original_to_openai_name[openai_name]
                    print(f"Tool called: {openai_name} (Original: {original_name})")
                    
                    # Create MCP-compatible tool call text
                    args_json = tc["function"].get("arguments", "{}")
                    mcp_text += f'<tool name="{original_name}" args=\'{args_json}\'/>\n'
        
        # Execute all tool calls through the processor
        if mcp_text:
            results = await processor.process_text(mcp_text)
            
            # Print results
            if results:
                print("=== Tool Results ===")
                for r in results:
                    dur = (r.end_time - r.start_time).total_seconds()
                    print(f"- {r.tool} ({dur:.3f}s):")
                    if r.error:
                        print(f"    [ERROR] {r.error}")
                    else:
                        print(f"    {r.result!r}")
            else:
                print("[no results returned]")
    elif not reply:
        print("[no response or tool calls]")

    # 6) Clean up
    await stream_mgr.close()

if __name__ == "__main__":
    asyncio.run(main())