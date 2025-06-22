#!/usr/bin/env python
# examples/mcp_round_trip.py
"""
MCP "round-trip" example with SQLite stdio transport using ToolProcessor directly.

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
import uuid
from typing import Any, Dict, List, Optional, Tuple

from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv

# CHUK Tool Processor imports
from chuk_tool_processor.mcp import setup_mcp_stdio
from chuk_tool_processor.registry import ToolRegistryProvider
from chuk_tool_processor.core.processor import ToolProcessor
from chuk_tool_processor.models.tool_result import ToolResult

# MCP CLI imports - only using llm_client and system_prompt_generator
from chuk_llm.llm.llm_client import get_llm_client
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator

# Initialize colorama for colored output
colorama_init(autoreset=True)

# ------------------------------------------------------------------ #
# Helper classes and functions for tool name adaptation
# ------------------------------------------------------------------ #
class ToolNameAdapter:
    """Handles adaptation between OpenAI-compatible tool names and MCP original names."""
    
    @staticmethod
    def to_openai_compatible(namespace: str, name: str) -> str:
        """Convert MCP tool name to OpenAI-compatible format."""
        return f"{namespace}_{name}"
    
    @staticmethod
    def from_openai_compatible(openai_name: str) -> str:
        """Convert OpenAI-compatible name back to MCP format."""
        if "_" in openai_name:
            parts = openai_name.split("_", 1)
            return f"{parts[0]}.{parts[1]}"
        return openai_name

def convert_to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tool metadata to OpenAI function format."""
    openai_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
            
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", "unknown"),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        })
    
    return openai_tools

def format_tool_response(response_content: Any) -> str:
    """Format tool response content for display."""
    if isinstance(response_content, (dict, list)):
        try:
            return json.dumps(response_content, indent=2)
        except:
            return str(response_content)
    return str(response_content)

# ------------------------------------------------------------------ #
# Helper functions for displaying registry tools and results
# ------------------------------------------------------------------ #
async def display_registry_tools(registry: Any, namespace_filter: Optional[str] = None) -> List[Tuple[str, str]]:
    """Display all tools in the registry, optionally filtered by namespace."""
    # Get tools, optionally filtered by namespace
    if namespace_filter:
        tools = [(ns, name) for ns, name in await registry.list_tools() if ns == namespace_filter]
    else:
        tools = await registry.list_tools()
    
    # Print tools
    print(Fore.CYAN + f"🔧  Registered MCP tools ({len(tools)}):")
    
    for ns, name in tools:
        md = await registry.get_metadata(name, ns)
        print(f"  • {Fore.GREEN}{ns}.{name:<20}{Style.RESET_ALL} - {md.description or '<no description>'}")
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
# Helper function for preparing OpenAI-compatible tools
# ------------------------------------------------------------------ #
async def prepare_openai_tools(registry: Any, tools: List[Tuple[str, str]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Prepare OpenAI-compatible tool definitions with name mapping.
    
    Args:
        registry: Tool registry
        tools: List of (namespace, name) tuples
        
    Returns:
        Tuple of (openai_tools, name_mapping)
    """
    tool_defs = []
    name_mapping = {}
    
    for ns, name in tools:
        md = await registry.get_metadata(name, ns)
        if md:
            openai_name = ToolNameAdapter.to_openai_compatible(ns, name)
            original_name = f"{ns}.{name}"
            name_mapping[openai_name] = original_name
            
            tool_defs.append({
                "name": openai_name,
                "description": f"{md.description or ''} (Original: {original_name})",
                "parameters": md.argument_schema or {}
            })
    
    openai_tools = convert_to_openai_tools(tool_defs)
    return openai_tools, name_mapping

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
        registry = await ToolRegistryProvider.get_registry()
        sqlite_tools = await display_registry_tools(registry, namespace_filter="sqlite")
        
        # 3) Create a ToolProcessor for handling tool execution
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
        openai_tools, name_mapping = await prepare_openai_tools(registry, sqlite_tools)
        
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
                    tool_call_id = tc.get("id") or f"call_{openai_name}_{uuid.uuid4().hex[:8]}"
                    
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