#!/usr/bin/env python
"""
MCP "round-trip" example with SQLite stdio transport using ToolProcessor.

This script demonstrates:
1. Connecting to an MCP SQLite server
2. Using CHUK ToolProcessor to handle tool execution
3. Handling OpenAI's function naming restrictions
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv

# CHUK Tool Processor imports
from chuk_tool_processor.mcp import setup_mcp_stdio
from chuk_tool_processor.registry import ToolRegistryProvider
from chuk_tool_processor.core.processor import ToolProcessor
from chuk_tool_processor.models.tool_result import ToolResult

# MCP CLI imports
from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator
from mcp_cli.llm.tools_handler import convert_to_openai_tools

# Initialize colorama for colored output
colorama_init(autoreset=True)

# ------------------------------------------------------------------ #
# Helper functions for displaying registry tools and results
# ------------------------------------------------------------------ #
def display_registry_tools(registry: Any, namespace_filter: Optional[str] = None) -> List:
    """Display all tools in the registry, optionally filtered by namespace."""
    # Get tools, optionally filtered by namespace
    if namespace_filter:
        tools = [(ns, name) for ns, name in registry.list_tools() if ns == namespace_filter]
    else:
        tools = registry.list_tools()
    
    # Print tools
    print(Fore.CYAN + f"🔧  Registered MCP tools ({len(tools)}):")
    
    for ns, name in tools:
        md = registry.get_metadata(name, ns)
        print(f"  • {Fore.GREEN}{ns}.{name:<20}{Style.RESET_ALL} – {md.description or '<no description>'}")
    print()
    
    return tools

def display_tool_results(results: List[ToolResult]) -> None:
    """Display tool execution results with nice formatting."""
    if not results:
        print(Fore.YELLOW + "[no results returned]")
        return
        
    print(Fore.CYAN + "=== Tool Results ===")
    
    for result in results:
        duration = (result.end_time - result.start_time).total_seconds()
        status = Fore.GREEN if not result.error else Fore.RED
        
        print(f"{status}{result.tool} ({duration:.3f}s){Style.RESET_ALL}")
        
        # Format arguments if available
        if hasattr(result, 'call') and result.call and hasattr(result.call, 'arguments'):
            print(f"  {Fore.YELLOW}Arguments:{Style.RESET_ALL} {json.dumps(result.call.arguments, indent=2)}")
        
        # Format result or error
        if result.error:
            print(f"  {Fore.RED}Error:{Style.RESET_ALL} {result.error}")
        else:
            if isinstance(result.result, (dict, list)):
                print(f"  {Fore.CYAN}Result:{Style.RESET_ALL} {json.dumps(result.result, indent=2)}")
            else:
                print(f"  {Fore.CYAN}Result:{Style.RESET_ALL} {result.result}")

# ------------------------------------------------------------------ #
# Helper functions for OpenAI tool name compatibility
# ------------------------------------------------------------------ #
def adapt_tool_names_for_openai(tools: List) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Adapt MCP tool names to be compatible with OpenAI's requirements.
    
    Note: This should ideally be part of the CHUK framework.
    """
    registry = ToolRegistryProvider.get_registry()
    mcp_tools = []
    original_to_openai_name = {}
    
    for ns, name in tools:
        md = registry.get_metadata(name, ns)
        if md:
            # Create OpenAI-compatible name (replace dots with underscores)
            openai_name = f"{ns}_{name}"
            original_name = f"{ns}.{name}"
            original_to_openai_name[openai_name] = original_name
            
            mcp_tools.append({
                "name": openai_name,
                "description": f"{md.description or ''} (Original: {original_name})",
                "parameters": md.argument_schema or {}
            })
    
    openai_tools = convert_to_openai_tools(mcp_tools)
    return openai_tools, original_to_openai_name

# ------------------------------------------------------------------ #
# Main function
# ------------------------------------------------------------------ #
async def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(description="MCP round-trip with ToolProcessor")
    p.add_argument("--provider", default="openai", help="LLM provider (openai|ollama)")
    p.add_argument("--model", default="gpt-4o-mini", help="Model name")
    p.add_argument("--prompt", required=True, help="User prompt for the LLM")
    args = p.parse_args()

    # 1) Set up MCP with the SQLite server
    processor, stream_mgr = await setup_mcp_stdio(
        config_file="server_config.json",
        servers=["sqlite"],
        server_names={0: "sqlite"},
        namespace="sqlite",
    )

    try:
        # 2) Get the registry and list tools
        registry = ToolRegistryProvider.get_registry()
        sqlite_tools = display_registry_tools(registry, namespace_filter="sqlite")
        
        # 3) Create a ToolProcessor for handling tool execution
        # This leverages the central processor class from CHUK
        tool_processor = ToolProcessor(
            registry=registry,
            default_timeout=10.0,
            max_concurrency=4,
            enable_caching=True,
            cache_ttl=300,
            enable_retries=True,
            max_retries=3
        )
        
        # 4) Prepare OpenAI-compatible tool definitions
        openai_tools, name_mapping = adapt_tool_names_for_openai(sqlite_tools)
        
        # 5) Send prompt to LLM
        client = get_llm_client(provider=args.provider, model=args.model)
        sys_prompt = SystemPromptGenerator().generate_prompt({"tools": openai_tools})
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": args.prompt},
        ]
        
        completion = await client.create_completion(messages=messages, tools=openai_tools)
        
        # Print the assistant's response
        reply = completion.get("response", "")
        tool_calls = completion.get("tool_calls", [])
        
        if reply:
            print(Fore.CYAN + "=== Assistant Reply ===")
            print(reply, end="\n\n")
        
        # 6) Process tool calls using the ToolProcessor
        if tool_calls:
            print(Fore.CYAN + "=== Tool Calls ===")
            
            # Process each tool call
            for tc in tool_calls:
                if tc.get("function") and "name" in tc.get("function", {}):
                    openai_name = tc["function"]["name"]
                    
                    # Convert back to original name for execution
                    if openai_name in name_mapping:
                        original_name = name_mapping[openai_name]
                        args_str = tc["function"].get("arguments", "{}")
                        args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                        
                        # Display tool call
                        print(f"{Fore.GREEN}Tool: {openai_name} → {original_name}{Style.RESET_ALL}")
                        print(f"  {Fore.YELLOW}Arguments:{Style.RESET_ALL} {json.dumps(args_dict, indent=2)}")
                        
                        # Format as XML for the processor
                        tool_call_text = f'<tool name="{original_name}" args=\'{json.dumps(args_dict)}\'/>'
                        
                        # Process text with the ToolProcessor
                        results = await tool_processor.process_text(tool_call_text)
                        
                        # Display results
                        display_tool_results(results)
                    else:
                        print(f"{Fore.RED}Unknown tool: {openai_name}{Style.RESET_ALL}")
        elif not reply:
            print(Fore.YELLOW + "[no response or tool calls]")
    
    finally:
        # 7) Clean up
        await stream_mgr.close()

if __name__ == "__main__":
    asyncio.run(main())